import argparse
import os
import sys
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SERVICE_ACCOUNT_FILE = "credentials.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def authenticate_drive_service_account(credentials_file):
    credentials = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
    return build("drive", "v3", credentials=credentials)


def list_folders(service, parent_id, drive_id=None):
    """List all subfolders of a given folder."""
    query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    params = {"q": query, "fields": "files(id, name)"}
    if drive_id:
        params.update(
            {
                "corpora": "drive",
                "driveId": drive_id,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }
        )
    result = service.files().list(**params).execute()
    return result.get("files", [])


def list_files(service, parent_id, drive_id=None):
    """List all files in a given folder."""
    query = f"'{parent_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
    params = {"q": query, "fields": "files(id, name)"}
    if drive_id:
        params.update(
            {
                "corpora": "drive",
                "driveId": drive_id,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }
        )
    result = service.files().list(**params).execute()
    return result.get("files", [])


def find_folder_by_name(service, name, parent_id, drive_id=None):
    """Find a folder by name under a parent folder."""
    query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    params = {"q": query, "fields": "files(id, name)"}
    if drive_id:
        params.update(
            {
                "corpora": "drive",
                "driveId": drive_id,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }
        )
    result = service.files().list(**params).execute()
    items = result.get("files", [])
    return items[0]["id"] if items else None


def find_top_level_folder(service, name, drive_id=None):
    """Find a top-level folder by name (in the root of a shared drive or My Drive)."""
    if drive_id:
        query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false and '{drive_id}' in parents"
        params = {
            "q": query,
            "fields": "files(id, name)",
            "corpora": "drive",
            "driveId": drive_id,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
    else:
        query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        params = {"q": query, "fields": "files(id, name)"}
    result = service.files().list(**params).execute()
    items = result.get("files", [])
    return items[0]["id"] if items else None


def download_file(service, file_id, file_name, dest_dir, drive_id=None):
    """Download a single file to dest_dir."""
    dest_path = Path(dest_dir) / file_name
    if dest_path.exists():
        print(f"  Skipping (already exists): {dest_path}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    params = {"fileId": file_id, "alt": "media"}
    if drive_id:
        params["supportsAllDrives"] = True

    request = service.files().get_media(fileId=file_id, supportsAllDrives=bool(drive_id))
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    print(f"  Downloaded: {dest_path}")
    return True


def download_date_folders(
    service,
    source_folder_name,
    local_base_dir,
    drive_id=None,
    date_filter=None,
    dry_run=False,
):
    """
    Download all date-named subfolders from a Google Drive folder to a local directory.

    Args:
        source_folder_name: Name of the top-level folder in Google Drive (e.g. 'ratings')
        local_base_dir: Local directory to download into (e.g. 'data/paper/github-actions/ratings')
        drive_id: Shared drive ID (optional)
        date_filter: Only download folders matching this date string (e.g. '2026-04-08')
        dry_run: If True, only print what would be downloaded
    """
    folder_id = find_top_level_folder(service, source_folder_name, drive_id)
    if not folder_id:
        print(f"Folder '{source_folder_name}' not found in Google Drive.")
        return 0

    date_folders = list_folders(service, folder_id, drive_id)
    date_folders.sort(key=lambda x: x["name"])

    if date_filter:
        date_folders = [f for f in date_folders if f["name"] == date_filter]
        if not date_folders:
            print(f"No folder matching date '{date_filter}' found under '{source_folder_name}'.")
            return 0

    total = 0
    for date_folder in date_folders:
        date = date_folder["name"]
        files = list_files(service, date_folder["id"], drive_id)
        print(f"  {source_folder_name}/{date}: {len(files)} file(s)")
        if dry_run:
            for f in files:
                print(f"    Would download: {f['name']}")
            continue
        for f in files:
            download_file(service, f["id"], f["name"], Path(local_base_dir) / date, drive_id)
            total += 1

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Download regression test data from Google Drive to local data/paper/github-actions/ structure"
    )
    parser.add_argument(
        "--credentials",
        "-c",
        default="credentials.json",
        help="Path to service account credentials file (default: credentials.json)",
    )
    parser.add_argument(
        "--drive-id",
        default=None,
        help="Google Drive shared drive ID (optional)",
    )
    parser.add_argument(
        "--local-base",
        default="data/paper/github-actions",
        help="Local base directory (default: data/paper/github-actions)",
    )
    parser.add_argument(
        "--date",
        "-d",
        default=None,
        help="Only download a specific date folder (e.g. 2026-04-08). Downloads all dates if omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--ratings-only",
        action="store_true",
        help="Only download ratings (skip dataset)",
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Only download dataset (skip ratings)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.credentials):
        print(f"Error: Credentials file '{args.credentials}' not found")
        return 1

    service = authenticate_drive_service_account(args.credentials)
    local_base = Path(args.local_base)

    if args.dry_run:
        print("DRY RUN — no files will be downloaded\n")

    total = 0

    if not args.ratings_only:
        print(f"Downloading dataset -> {local_base}/dataset/")
        total += download_date_folders(
            service,
            source_folder_name="dataset",
            local_base_dir=local_base / "dataset",
            drive_id=args.drive_id,
            date_filter=args.date,
            dry_run=args.dry_run,
        )

    if not args.dataset_only:
        print(f"Downloading ratings -> {local_base}/ratings/")
        total += download_date_folders(
            service,
            source_folder_name="ratings",
            local_base_dir=local_base / "ratings",
            drive_id=args.drive_id,
            date_filter=args.date,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        print(f"\nDone. {total} file(s) downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
