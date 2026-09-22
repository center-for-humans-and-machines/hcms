import argparse
import glob
import os
import sys
from datetime import datetime

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Path to your service account key file
SERVICE_ACCOUNT_FILE = "credentials.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]


def authenticate_drive_service_account():
    """Authenticate using service account file."""
    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=credentials)
    return service


def find_folder_by_path(service, folder_name, parent_folder_id=None, drive_id=None):
    """Check if a folder exists in Google Drive.

    Args:
        service: Authenticated Google Drive service
        folder_name: Name of the folder to check
        parent_folder_id: ID of parent folder (optional, None for root)
        drive_id: Google Drive ID for shared drives (optional)

    Returns:
        dict: Folder info if exists, None if not found
    """
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"

    if parent_folder_id:
        query += f" and '{parent_folder_id}' in parents"

    list_params = {
        "q": query,
        "fields": "files(id, name)",
    }

    if drive_id:
        list_params.update(
            {
                "corpora": "drive",
                "driveId": drive_id,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }
        )

    results = service.files().list(**list_params).execute()
    items = results.get("files", [])

    if items:
        print("Files:")
        for item in items:
            print(f"{item['name']} ({item['id']})")

    if not items:
        print(f"Folder '{folder_name}' not found.")
        return None

    return items[0]["id"]


def create_folder(service, folder_name, parent_folder_id, drive_id=None):
    """Create a new folder in Google Drive."""
    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }

    create_params = {"body": folder_metadata, "fields": "id"}

    if drive_id:
        create_params.update(
            {
                "supportsAllDrives": True,
            }
        )

    folder = service.files().create(**create_params).execute()
    print(f"Created folder '{folder_name}' with ID: {folder.get('id')}")
    return folder.get("id")


def upload_file(service, file_path, folder_id, drive_id=None):
    """Upload a single file to the specified folder."""
    file_name = os.path.basename(file_path)

    # Determine MIME type based on file extension
    file_extension = os.path.splitext(file_name)[1].lower()
    mime_type_map = {
        ".json": "application/json",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    mime_type = mime_type_map.get(file_extension, "application/octet-stream")

    file_metadata = {"name": file_name, "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype=mime_type)

    create_params = {"body": file_metadata, "media_body": media, "fields": "id"}

    if drive_id:
        create_params.update(
            {
                "supportsAllDrives": True,
            }
        )

    try:
        file = service.files().create(**create_params).execute()
        print(f"Successfully uploaded '{file_name}' with ID: {file.get('id')}")
        return file.get("id")
    except Exception as e:
        print(f"Failed to upload '{file_name}': {str(e)}")
        return None


def upload_files_by_pattern(file_pattern, target_path="github-actions"):
    """
    Upload files matching a pattern to a date-named folder in Google Drive.

    Args:
        file_pattern (str): File pattern to match (e.g., "data/dataset/*.json")
        target_path (str): Path in Google Drive where the date folder will be created
    """
    # Authenticate
    service = authenticate_drive_service_account()

    # Find the target folder
    target_folder_id = find_folder_by_path(service, target_path)
    if not target_folder_id:
        print(f"Target path '{target_path}' not found. Please create it first.")
        return

    # Create today's date folder
    today = datetime.now().strftime("%Y-%m-%d")
    date_folder_id = create_folder(service, today, target_folder_id)

    # Find all files matching the pattern
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    print(f"Found {len(matching_files)} files matching pattern: {file_pattern}")

    # Upload each file
    successful_uploads = 0
    for file_path in matching_files:
        if os.path.isfile(file_path):
            file_id = upload_file(service, file_path, date_folder_id)
            if file_id:
                successful_uploads += 1
        else:
            print(f"Skipping '{file_path}' - not a file")

    print(
        f"\nUpload complete: {successful_uploads}/{len(matching_files)} files uploaded successfully"
    )
    print(f"Files uploaded to: {target_path}/{today}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload files matching a pattern to Google Drive with date-based folder organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gdrive_upload.py "data/dataset/*.json"
  python gdrive_upload.py "logs/*.txt" --target-path "github-actions" --credentials "my_creds.json"
  python gdrive_upload.py "data/*.csv" --date-folder "2024-12-25"
  python gdrive_upload.py "images/*.jpg" --no-date-folder --target-path "hcms/images"
        """,
    )

    parser.add_argument(
        "file_pattern",
        help="File pattern to match (e.g., 'data/dataset/*.json', 'reports/*.pdf')",
    )

    parser.add_argument(
        "--target-path",
        "-t",
        default="github-actions",
        help="Target path in Google Drive (default: github-actions)",
    )

    parser.add_argument(
        "--credentials",
        "-c",
        default="credentials.json",
        help="Path to service account credentials file (default: credentials.json)",
    )

    parser.add_argument(
        "--date-folder",
        "-d",
        default=None,
        help="Custom date folder name (default: today's date as YYYY-MM-DD). Use 'none' for no date folder.",
    )

    parser.add_argument(
        "--no-date-folder",
        action="store_true",
        help="Upload directly to target path without creating a date folder",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--drive-id",
        default=None,
        help="Google Drive ID for shared drives (optional, omit for regular Drive access)",
    )

    return parser.parse_args()


def upload_files_by_pattern(
    file_pattern,
    target_path="github-actions",
    credentials_file="credentials.json",
    date_folder=None,
    no_date_folder=False,
    dry_run=False,
    verbose=False,
    drive_id=None,
):
    """
    Upload files matching a pattern to a date-named folder in Google Drive.

    Args:
        file_pattern (str): File pattern to match (e.g., "data/dataset/*.json")
        target_path (str): Path in Google Drive where the date folder will be created
        credentials_file (str): Path to service account credentials file
        date_folder (str): Custom date folder name, or None for today's date
        no_date_folder (bool): If True, upload directly to target path
        dry_run (bool): If True, show what would be uploaded without uploading
        verbose (bool): Enable verbose output
    """
    global SERVICE_ACCOUNT_FILE
    SERVICE_ACCOUNT_FILE = credentials_file

    if verbose:
        print(f"Using credentials file: {credentials_file}")
        print(f"File pattern: {file_pattern}")
        print(f"Target path: {target_path}")

    # Find all files matching the pattern
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"No files found matching pattern: {file_pattern}")
        return 1

    print(f"Found {len(matching_files)} files matching pattern: {file_pattern}")
    if verbose:
        for file_path in matching_files:
            print(f"  - {file_path}")

    if dry_run:
        print("DRY RUN - No files will be uploaded")
        if no_date_folder:
            print(f"Would upload to: {target_path}")
        else:
            date_str = (
                date_folder if date_folder else datetime.now().strftime("%Y-%m-%d")
            )
            print(f"Would upload to: {target_path}/{date_str}")
        return 0

    # Authenticate
    try:
        service = authenticate_drive_service_account()
        if verbose:
            print("Successfully authenticated with Google Drive")
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        return 1

    # Find the target folder
    target_folder_id = find_folder_by_path(service, target_path, drive_id=drive_id)
    if not target_folder_id:
        print(f"Target path '{target_path}' not found. Please create it first.")
        return 1

    # Determine upload folder
    if no_date_folder:
        upload_folder_id = target_folder_id
        upload_path = target_path
    else:
        # Create date folder
        if date_folder:
            date_str = date_folder
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")

        upload_folder_id = get_or_create_folder(
            service, date_str, target_folder_id, drive_id
        )
        upload_path = f"{target_path}/{date_str}"

    # Upload each file
    successful_uploads = 0
    for file_path in matching_files:
        if os.path.isfile(file_path):
            file_id = upload_file(service, file_path, upload_folder_id, drive_id)
            if file_id:
                successful_uploads += 1
        else:
            print(f"Skipping '{file_path}' - not a file")

    print(
        f"\nUpload complete: {successful_uploads}/{len(matching_files)} files uploaded successfully"
    )
    print(f"Files uploaded to: {upload_path}")

    return 0 if successful_uploads == len(matching_files) else 1


def get_or_create_folder(service, folder_name, parent_folder_id, drive_id=None):
    """Get existing folder or create it if it doesn't exist."""
    # First check if folder already exists
    existing_folder_id = find_folder_by_path(
        service, folder_name, parent_folder_id, drive_id
    )

    if existing_folder_id:
        print(f"Using existing folder '{folder_name}' with ID: {existing_folder_id}")
        return existing_folder_id

    # Create folder if it doesn't exist
    return create_folder(service, folder_name, parent_folder_id, drive_id)


def main():
    """Main function to handle command line execution."""
    args = parse_arguments()

    # Validate arguments
    if not os.path.exists(args.credentials):
        print(f"Error: Credentials file '{args.credentials}' not found")
        return 1

    # Handle date folder logic
    date_folder = None
    no_date_folder = args.no_date_folder

    if args.date_folder:
        if args.date_folder.lower() == "none":
            no_date_folder = True
        else:
            date_folder = args.date_folder

    # Call the upload function
    return upload_files_by_pattern(
        file_pattern=args.file_pattern,
        target_path=args.target_path,
        credentials_file=args.credentials,
        date_folder=date_folder,
        no_date_folder=no_date_folder,
        dry_run=args.dry_run,
        verbose=args.verbose,
        drive_id=args.drive_id,
    )


# Example usage
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
