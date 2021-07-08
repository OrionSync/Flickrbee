# Flickrbee Shannon Berry commissioned by Robert Moore
# Copyright 2021, MIT License

import confuse
import logging
import logging.config
import flickrapi
import xml.etree.ElementTree as ET
import json
import os
import re
import pandas as pd
import requests
import cv2
import numpy as np
import time
import tensorflow as tf
from PIL import Image
from io import BytesIO
from hashlib import sha256
from datetime import datetime


class Flickrbee:
    def __init__(self):
        self.config = confuse.Configuration('Flickrbee', __name__)
        self.config.set_file('config/flickrbee_config.yaml')
        logging.config.fileConfig('config/logging.properties')

        self.config_postcode = self.config['postcode']['default'].get()
        self.config_filtered = self.config['download']['filtered'].get()
        self.config_grayscale = self.config['download']['grayscale'].get()
        self.config_prefix = self.config['image']['prefix'].get()

        self.flickr = flickrapi.FlickrAPI(
            self.config['FLICKR']['Id'].get(),
            self.config['FLICKR']['Secret'].get(),
            format='etree', cache=True)

        exportspath = ['exports/', 'exports/csv', 'exports/filtered-images', 'exports/original-images']
        for path in exportspath:
            if not os.path.exists(path):
                os.mkdir(path)


    def load_postcodes(self):
        with open('postcodes/australian_postcodes.json') as f:
            postcodes = json.load(f)
            return postcodes

    def get_description(self, _input):
        xmltostring = ET.tostring(_input, encoding='unicode')
        desc = re.findall(r'(?<=<description>).*?(?=</description>)', xmltostring)
        if desc:
            return desc[0]
        else:
            return None

    def get_postcode(self, title, tags, desc):
        default = self.config_postcode
        found = re.findall('(?<!\d)[0-7]\d{3}(?!\d)', title)
        if found:
            return found[0]
        else:
            found = re.findall('(?<!\d)[0-7]\d{3}(?!\d)', tags)
            if found:
                return found[0]
            else:
                if type(desc) is str:
                    found = re.findall('(?<!\d)[0-7]\d{3}(?!\d)', desc)
                    if found:
                        return found[0]
                else:
                    return default

    def location_data(self, postcode):
        data_pool = self.load_postcodes()
        try:
            if postcode == '0000':
                locality = "No Data Present"
                return locality
            else:
                entry = next(filter(lambda x: x['postcode'] == postcode, data_pool))
                locality = entry['locality']
                return locality
        except StopIteration:
            print(f'No Locality value was found for {postcode}')
            locality = 'No Data Present'
            return locality

    def create_dataframe(self, collection):
        df = pd.DataFrame(collection, columns=['Date', 'Title', 'Postcode', 'Locality', 'Url'])
        return df

    def download(self, original, filtered):
        if self.config_filtered is True:
            url_list = filtered
            save_path = 'exports/filtered-images'
        else:
            url_list = original
            save_path = 'exports/original-images'

        for url in url_list:
            path = save_path
            response = requests.get(url)
            h = sha256(response.content).hexdigest()
            if self.config_grayscale is True:
                orig_img = Image.open(BytesIO(response.content))
                np_image = np.array(orig_img)
                img = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
                p = os.path.join(path, f"{self.config_prefix}-{h}-GRAYSCALE.{'jpg'}")
                if not os.path.exists(p):
                    cv2.imwrite(f"{path}/{self.config_prefix}-{h}-GRAYSCALE.{'jpg'}", img)
                else:
                    print(f'{p} already exists --Skipped--')

            else:
                img = Image.open(BytesIO(response.content))

                p = os.path.join(path, f"{self.config_prefix}-{h}.{'jpg'}")
                if not os.path.exists(p):
                    img.save(p)
                else:
                    print(f'{p} already exists --Skipped--')

    def export(self, collection):
        df = self.create_dataframe(collection)
        df.to_csv(r'exports/csv/Flickr-bee-data.csv', index=False, header=True)
        original_urls = df['Url'].tolist()
        filtered_df = df[df.Postcode != '0000']
        filtered_df.to_csv(r'exports/csv/Flickr-bee-data-FILTERED.csv', index=False, header=True)
        filtered_urls = filtered_df['Url'].tolist()
        logging.info('Flickrbee Project exports COMPLETE...')
        logging.info('Initializing Image Download...')
        self.download(original_urls, filtered_urls)
        return original_urls, filtered_urls

    def run(self):
        start_time = time.time()
        logging.info("Initializing Flickrbee Project...")
        collection = []

        photos = self.flickr.walk(
            text='#beesathome',
            tag_mode='all',
            privacy_filter=1,
            perpage=100,
            sort='relevance',
            extras=['url_c, tags, geo, description, date_taken'],
            min_upload_date='2020-09-01',
            max_upload_date='2021-01-15')

        for photo in photos:
            title = photo.get('title')
            date = photo.get('datetaken')
            tags = photo.get('tags')
            url = photo.get('url_c')
            desc = self.get_description(photo)
            postcode = self.get_postcode(title, tags, desc)
            locality = self.location_data(postcode)
            collection.append([date, title, postcode, locality, url])

        self.export(collection)
        end_time = start_time - time.time()
        print(end_time)


class ImageDetection:
    def __init__(self):
        self.config = confuse.Configuration('Flickrbee', __name__)
        self.config.set_file('config/flickrbee_config.yaml')
        logging.config.fileConfig('config/logging.properties')

        self.config_model = self.config['TENSOR']['Model']
        self.config_labels = self.config['TENSOR']['Labels']
        self.config_filtered = self.config['download']['filtered']
        self.config_imagedir = self.config['TENSOR']['ImageDir']
        self.config_imagedirfiltered = self.config['TENSOR']['ImageDirFiltered']
        self.config_threshold = self.config['TENSOR']['Threshold']

    def imageDir(self):
        if self.config_filtered:
            return self.config_imagedirfiltered
        else:
            return self.config_imagedir

    def tensorflow(self):
        tf.get_logger().setLevel('ERROR')  

        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        
        IMAGEDIR = self.imageDir()
        DIRECTORIES = ['exported-csv', 'exported-images']

        import time
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils

        for i in DIRECTORIES:
            for filename in os.listdir(i):
                filePath = os.path.join(i, filename)
                try:
                    if os.path.isfile(filePath):
                        os.remove(filePath)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (filePath, e))

        print('Loading model...', end='')
        start_time = time.time()

        
        detect_fn = tf.saved_model.load(self.config_model)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))

        

        category_index = label_map_util.create_category_index_from_labelmap(self.config_labels,
                                                                            use_display_name=True)

        import numpy as np
        import warnings
        warnings.filterwarnings('ignore')

        exported_list = []

        for filename in os.listdir(IMAGEDIR):
            print('Running inference for {}... '.format(filename), end='')
            print(os.path.join(IMAGEDIR, filename))
            image = cv2.imread(os.path.join(IMAGEDIR, filename))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_expanded = np.expand_dims(image_rgb, axis=0)

            input_tensor = tf.convert_to_tensor(image)

            input_tensor = input_tensor[tf.newaxis, ...]

            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            image_with_detections = image.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=self.config_threshold,
                agnostic_mode=False)

            print('Done')
            keys = list(detections.keys())
            print(keys)
            score = detections.get('detection_scores')

            scorelist = []
            for i in score:
                if i >= self.config_threshold:
                    scorelist.append(round(100 * i))

            exported_list.append([filename, len(scorelist), scorelist])
            df = pd.DataFrame(exported_list, columns=['Title', 'Total', 'Scores (%)'])
            cv2.imwrite(
                f'/exports/detected-images/{filename}',
                image_with_detections)
            df.to_csv(r'exported-csv/exported' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.csv')


task = Flickrbee()
task.run()
