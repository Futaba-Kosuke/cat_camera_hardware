import os
import datetime
import platform

import firebase_admin
from firebase_admin import credentials, storage, firestore

class Firebase:

    def __init__(self, cred_path='./firebase/cred.json', bucket_name='cat-camera-cafe8.appspot.com'):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})

        self.bucket_name = bucket_name
        self.bucket = storage.bucket()

        self.db = firestore.client()

    def upload_file(self, file_path='./firebase/image.jpeg', content_type='image/jpeg', meta_data={'from': platform.system()}):
        blob = self.bucket.blob(file_path.split('/')[-1])
        with open(file_path, 'rb') as f:
            blob.upload_from_file(f, content_type=content_type)
        blob.metadata = meta_data
        blob.patch()

        blob.make_public()
        url = blob.public_url
        self.db.collection("images").add({
            'isFavorite': False,
            'url': url,
            'postTime': datetime.datetime.now()
        })

        os.remove(file_path)
