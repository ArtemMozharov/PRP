import torch
from settings import *
from prp import *
from preprocess import *
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from model import construct_PPNet

### Load trained ProtoPNet model
load_model_path = os.path.join(load_model_dir, load_model_name)
state_dict = torch.load(load_model_path,map_location=torch.device(device))

ppnet = construct_PPNet(
    base_architecture=base_architecture,
    img_size=img_size,
    prototype_shape=prototype_shape, 
    prototype_activation_function=prototype_activation_function,
    add_on_layers_type=add_on_layers_type,
)

ppnet.load_state_dict(state_dict)
ppnet = ppnet.to(device)
ppnet.eval()

prp_model = PRPCanonizedModel(ppnet,base_architecture)



### For a single test image and for a particular prototype
test_image_path = os.path.join(test_image_dir, test_image_name)
img = torch.asarray(np.array(Image.open(test_image_path)).transpose([2,0,1])).float()
img_variable = Variable(img.unsqueeze(0))/255
img_tensor = preprocess(img_variable,mean,std)
images_test = img_tensor.to(device)
prp_map = generate_prp_image(images_test, prototype_number, prp_model, device)
makedir(write_path)
plt.imsave(write_path+"prp_"+str(prototype_number)+"_"+test_image_name, prp_map, cmap="seismic", vmin=-1, vmax=+1)



#############
### OVERLAY ##########
################
def invert_normalize(ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    print(ten.shape)
    s = torch.tensor(np.asarray(std, dtype=np.float32)).unsqueeze(1).unsqueeze(2)
    m = torch.tensor(np.asarray(mean, dtype=np.float32)).unsqueeze(1).unsqueeze(2)

    res = ten * s + m
    return res

heatmap = cv2.imread(test_image_path)
heatmap = heatmap[..., ::-1]
heatmap = np.float32(heatmap) / 255
ts = invert_normalize(img_tensor.squeeze())
a = ts.data.numpy().transpose((1, 2, 0))
overlayed_original_img_j = 0.2 * a + 0.6 * heatmap
plt.imsave("Test images/"+"Overlay_prp_"+str(prototype_number)+"_"+test_image_name,
           overlayed_original_img_j,
           vmin=-1,
           vmax=+1.0)



