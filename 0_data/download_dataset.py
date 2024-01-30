import os
import gdown
import zipfile

def download_file_from_google_drive(id, destination):
    url = 'https://drive.google.com/uc?id={}'.format(id)

    gdown.download(url, destination, quiet=False)

def extract_zip(zip_path, destination_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

if not os.path.isdir('0_data'):
    os.mkdir('0_data')

print('Downloading Camera 1 Example data...')
download_file_from_google_drive('1qES6teQUprs-cgL-nACzFighjm_Dho0L', '0_data/Cam1.zip')

print('Downloading Camera 2 Example data...')
download_file_from_google_drive('1wiz5XPricNj-sFwqfj7tVI1IomlyO6xM', '0_data/Cam2.zip')

print('Downloading Outdoor Example data...')
download_file_from_google_drive('1UFah1QWltNDzUlsQ4HCMnuiyAAUh-9EE', '0_data/Outdoor.zip')

print('Extracting ZIP files...')
extract_zip('0_data/Cam1.zip', '0_data')
extract_zip('0_data/Cam2.zip', '0_data')
extract_zip('0_data/Outdoor.zip', '0_data')
