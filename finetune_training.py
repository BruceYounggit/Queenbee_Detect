import time

import PIL.Image
from ultralytics import YOLO

# Load a model
model = YOLO("model_weight/yolov8_pretrained_model/yolov8n.pt")

# train the model
model.train(data='config/QueenBees.yaml', epochs=500, batch=7, workers=4, patience=150, visualize=True,
            augment=True)  # train the model

# single test
results = model('imgs/img.png')
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = PIL.Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    current_time = time.time()
    time_struct = time.localtime(current_time)
    no_space_time_str = time.strftime("%Y%m%d%H%M", time_struct)
    im.save(f'results_{no_space_time_str}.jpg')  # save image
