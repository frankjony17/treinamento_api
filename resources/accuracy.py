
class Accuracy(object):

    def __init__(self):
        self.start_learning = None
        self.stop_learning = None
        self.elapsed_learning = None
        self.classification_report_classifier = None
        self.accuracy = None
        self.confusion_matrix = None
        self.filename = None

    def mapper(self, model):
        model['start_learning'] = str(self.start_learning)
        model['stop_learning'] = str(self.stop_learning)
        model['elapsed_learning'] = str(self.elapsed_learning)
        # model['classification_report_classifier'] = str(self.classification_report_classifier).\
        #     replace('\n', ' ').\
        #     replace('              ', ' ').\
        #     replace('         ', ' ').\
        #     replace('       ', ' ').\
        #     replace('      ', ' ').\
        #     replace('    ', ' ').\
        #     replace('  ', ' ').\
        #     replace('           ', ' ')
        model['accuracy'] = str(self.accuracy)
        model['confusion_matrix'] = str(self.confusion_matrix).replace('\n', '')
        model['filename'] = str(self.filename)
        return model

    def __str__(self):
        s = ""
        s += "start_learning:" + self.start_learning + ", "
        s += "stop_learning: " + str(self.stop_learning) + ", "
        s += "elapsed_learning: " + str(self.elapsed_learning) + ", "
        # s += "classification_report_classifier: " + str(self.classification_report_classifier) + ", "
        s += "accuracy: " + str(self.accuracy) + ", "
        s += "confusion_matrix: " + str(self.confusion_matrix).strip() + ", "
        # s += "filename: " + str(self.filename)
        s += ""
        return s.strip()

