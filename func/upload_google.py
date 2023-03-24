############################################################################################################
#   upload to google drive (https://console.cloud.google.com/welcome?project=prefab-gift-372805&hl=zh-tw)  #
############################################################################################################
import datetime
import os
import pickle
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request


def Create_Service(client_secret_file, api_name, api_version, logger, *scopes):
    logger.info(f"{client_secret_file}-{api_name}-{api_version}-{scopes}")
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes[0]]

    cred = None

    pickle_file = f"token_{API_SERVICE_NAME}_{API_VERSION}.pickle"

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            cred = flow.run_local_server()

        with open(pickle_file, "wb") as token:
            pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        logger.info(
            f"{str(datetime.datetime.now().astimezone(datetime.timezone(datetime.timedelta(hours=8))))} : {API_SERVICE_NAME} service created successfully"
        )
        return service
    except Exception as e:
        logger.debug(
            f"{str(datetime.datetime.now().astimezone(datetime.timezone(datetime.timedelta(hours=8))))} : Unable to connect."
        )
        logger.debug(e)
        return None


def upload_xlsx_to_google(filename, folder_id, logger):
    logger.info(
        f"{str(datetime.datetime.now().astimezone(datetime.timezone(datetime.timedelta(hours=8))))} : uploading file to google drive"
    )

    CLIENT_SECRET_FILE = "credentials.json"
    API_NAME = "drive"
    API_VERSION = "v3"
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, logger, SCOPES)

    file_metadata = {"name": filename, "parents": [folder_id]}

    media = MediaFileUpload(
        filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    service.files().create(body=file_metadata, media_body=media, fields="id").execute()
