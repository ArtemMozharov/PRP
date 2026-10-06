
load_model_dir = "../experiments/models/"
load_model_name ="protopnet_cub_best.pth"

test_image_dir = "Test images/"
test_image_name = "1435_stop_1331867282.avi_image2.png" 

write_path = "../experiments/runs/protopnet/"

prototype_number = 31

base_architecture = 'resnet34'
img_size = 224
prototype_shape = (2000, 128, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
