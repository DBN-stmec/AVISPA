from classify.engines.keras import KerasEngine
import datetime
import config
import stats
import logging
import os


class Classifier:
    model_name = None

    def __init__(self, model_name=config.CLASSIFY['model'], engine=config.CLASSIFY['engine']):
        self.model_name = model_name
        self.model_type = engine
        self.warmed_up = False
        self.engine = None
        self.init_engine()

    def init_engine(self):
        if self.model_type == "keras":
            self.engine = KerasEngine(model_name=self.model_name)
        else:
            exit('Invalid engine')

    def get_input_size(self):
        return self.engine.get_input_size()

    def process_file(self, file):
        if not self.warmed_up:
            logging.debug("Warming up")
            self.engine.process_file(file)
            stats.set('y_score', [])
            stats.set('y_pred', [])
            self.warmed_up = True

        classification_start = datetime.datetime.now()
        result = self.engine.process_file(file)
        classification_time = datetime.datetime.now() - classification_start
        stats.append("classification_time", classification_time.total_seconds() * 1000)

        return result

    def process_image(self, image_np):
        return self.engine.process_image(image_np)

    def process_and_mark_image(self, image_np):
        return self.engine.process_and_mark_image(image_np)

    def train(self, train_dir, val_dir, nb_epochs, batch_size):
        return self.engine.train(train_dir=train_dir, val_dir=val_dir, nb_epochs=nb_epochs, batch_size=batch_size)

    def exit(self):
        try:
            os.remove('tmp.jpg')
        except FileNotFoundError:
            pass
        self.engine.exit()
