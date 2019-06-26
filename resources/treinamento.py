import numpy as np
import glob
import cv2
import datetime as dt
import pickle

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from resources.accuracy import Accuracy


class Treinamento(object):
    def __init__(self, model):
        self.bool = True
        original_image_path = model['endereco'] + '/original_images/*.jpg'
        spoofing_image_path = model['endereco'] + '/spoofing_images/*.jpg'
        # Open the images
        self.original_image_list = self.open_image(original_image_path)
        self.spoofing_image_list = self.open_image(spoofing_image_path)

        if len(self.original_image_list) > 0 and len(self.spoofing_image_list) > 0:
            # Image processing algorithms
            or_image_list = self.apply_filters(self.original_image_list)
            sp_image_list = self.apply_filters(self.spoofing_image_list)
            # Creating the dataset
            dataset = self.dataset(or_image_list, sp_image_list)
            self.data = dataset['data']
            self.data_norm = dataset['data_norm']
            # Dividing the data set
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.data_norm,
                self.spoofing_images(),
                test_size=0.2
            )
            # SVM Classivier parameters
            self.classifier = self.classifier_parameters()
            # return json object
            self.json_report = Accuracy()
        else:
            self.bool = False

    @staticmethod
    def open_image(img_path):
        image_list = []
        for filename in glob.glob(img_path):
            imgage = cv2.imread(filename)
            image_list.append(imgage)
        return image_list

    @staticmethod
    def resize(image):
        y = cv2.resize(image, (200, 300))
        return y

    @staticmethod
    def anisotropic_diffusion(image):
        y = cv2.ximgproc.anisotropicDiffusion(image, 0.8, 5, 50)
        return y

    @staticmethod
    def diff_filter(image, image_filter):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(image_filter, cv2.COLOR_BGR2GRAY)

    def apply_filters(self, img_list):
        image_list = []
        for i in range(len(img_list)):
            img = self.resize(img_list[i])
            image = self.anisotropic_diffusion(img)
            image = self.diff_filter(img, image)
            image_list.append(image)
        return image_list

    # Creating the dataset
    @staticmethod
    def dataset(or_image_list, sp_image_list):
        data = np.concatenate((or_image_list, sp_image_list), axis=0)
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        data_norm = []
        for i in range(len(data)):
            data_norm.append(data[i] / 255.0)
        data_norm = np.array(data_norm)
        return {
            'data_norm': data_norm,
            'data': data
        }

    # Number of spoofing images and normal images must be equal
    def spoofing_images(self):
        y1 = np.zeros(self.data.shape[0] // 2)
        y2 = np.ones(self.data.shape[0] // 2)
        y = np.concatenate((y1, y2), axis=0).astype(int)
        return y

    # SVM Classivier parameters
    @staticmethod
    def classifier_parameters():
        return svm.SVC(
            C=5.0,
            cache_size=248,
            coef0=0.0,
            degree=30,
            gamma='scale',
            kernel='linear',
            max_iter=- 1,
            shrinking=True,
            tol=0.01,
            probability=True,
            verbose=False
        )

    # Learning
    def learning(self):
        start_time = dt.datetime.now()
        self.json_report.start_learning = f'{start_time:%Y-%m-%d %H:%M:%S%z}'
        self.classifier.fit(self.x_train, self.y_train)
        end_time = dt.datetime.now()
        self.json_report.stop_learning = f'{end_time:%Y-%m-%d %H:%M:%S%z}'
        elapsed_time = end_time - start_time
        self.json_report.elapsed_learning = str(elapsed_time)

    # Testing the model
    def testing_model(self):
        expected = self.y_test
        predicted = self.classifier.predict(self.x_test)
        self.json_report.classification_report_classifier = \
            "%s:\n%s\n" % (self.classifier, metrics.classification_report(expected, predicted))
        cm = metrics.confusion_matrix(expected, predicted)
        self.json_report.accuracy = metrics.accuracy_score(expected, predicted)
        self.json_report.confusion_matrix = cm

    # Save the model
    def save_model(self):
        filename = 'model/svm_model_anisotropic.pickle'
        self.json_report.filename = filename.replace('model/', '')
        pickle.dump(self.classifier, open(filename, 'wb'))
        return self.json_report
