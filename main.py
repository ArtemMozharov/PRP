import torch
from prp import *
from preprocess import *
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from model import construct_PPNet
from config import load_config
from settings import config_path, experiments_path
from tqdm import tqdm
from pathlib import Path



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config=load_config(config_path, experiments_path)


for experiment in tqdm(config.experiments, desc="Experiments"):

    image_dir = Path(experiment.test_image_dir)
    output_dir = Path(experiment.write_path)

    # Only process image files
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        print(f"No images found in {image_dir}")
        continue

    # Load the model ONCE per experiment
    load_model_path = os.path.join(
        experiment.load_model_dir,
        experiment.load_model_name
    )

    state_dict = torch.load(
        load_model_path,
        map_location=device
    )

    if experiment.model == "protopnet":
        ppnet = construct_PPNet(
            base_architecture=experiment.base_architecture,
            img_size=experiment.img_size,
            prototype_shape=experiment.prototype_shape,
            prototype_activation_function=(
                experiment.prototype_activation_function
            ),
            add_on_layers_type=experiment.add_on_layers_type,
        )

        ppnet.load_state_dict(state_dict)
        ppnet = ppnet.to(device)
        ppnet.eval()

        prp_model = PRPCanonizedModel(
            ppnet,
            experiment.base_architecture
        )
    else:
        raise NotImplementedError(
            f"Unsupported model: {experiment.model}"
        )

    # Iterate over every image
    for test_image_path in tqdm(
        image_paths,
        desc=f"Images: {experiment.model}",
        leave=False
    ):

        # Convert to RGB and resize
        with Image.open(test_image_path) as image:
            pil_img = image.convert("RGB").resize(
                (experiment.img_size, experiment.img_size)
            )

        img = torch.from_numpy(
            np.array(pil_img).transpose(2, 0, 1).copy()
        ).float()

        img_variable = img.unsqueeze(0) / 255.0
        img_tensor = preprocess(img_variable, mean, std)
        images_test = img_tensor.to(device)

        # One output directory per image
        image_output_dir = output_dir / test_image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Get the predicted class for this image
        with torch.no_grad():
            logits, min_distances = ppnet(images_test)
            predicted_class = logits.argmax(dim=1).item()
            # Keep the spatial prototype activations for visualization. The
            # model's forward pass returns only globally pooled distances.
            _, prototype_distances = ppnet.push_forward(images_test)
            similarities = ppnet.distance_2_similarity(prototype_distances)


        # Find all prototypes assigned to the predicted class
        prototype_ids = torch.where(
            ppnet.prototype_class_identity[:, predicted_class] == 1
        )[0].tolist()

        print(
            f"Image: {test_image_path.name} | "
            f"Predicted class: {predicted_class} | "
            f"Relevant prototypes: {len(prototype_ids)}"
        )

        for prototype_number in tqdm(prototype_ids,
            desc=f"Prototypes: {test_image_path.name}",
            leave=False
        ):

            # Generate PRP relevance map
            prp_map = generate_prp_image(
                images_test,
                prototype_number,
                prp_model,
                device
            )

            prp_map = np.asarray(prp_map).squeeze()

            # Save all outputs for this prototype in its own directory.
            prototype_output_dir = image_output_dir / f"prototype_{prototype_number}"
            prototype_output_dir.mkdir(parents=True, exist_ok=True)

            prp_heatmap_path = prototype_output_dir / f"prp_heatmap_{test_image_path.stem}.png"
            # Older runs may have created the image filename as a directory.
            # Remove that mistaken directory only when it is empty.
            if prp_heatmap_path.is_dir() and not any(prp_heatmap_path.iterdir()):
                prp_heatmap_path.rmdir()

            plt.imsave(
                prp_heatmap_path,
                prp_map,
                cmap="seismic",
                vmin=-1,
                vmax=1
            )

            # Generate overlay of PRP heatmap on original image
            original = np.asarray(pil_img).astype(np.float32) / 255.0

            # Map relevance values from [-1, 1] to [0, 1]
            normalized_map = np.clip(
                (prp_map - (-1.0)) / 2.0,
                0.0,
                1.0
            )

            heatmap = plt.get_cmap("seismic")(
                normalized_map
            )[..., :3]

            overlay = (
                0.6 * original + 0.4 * heatmap
            ).clip(0, 1)

            prp_overlay_path = prototype_output_dir / f"prp_overlay_{test_image_path.stem}.png"
            plt.imsave(prp_overlay_path, overlay)

            # Get the spatial similarity map for this prototype
            activation_map = similarities[0, prototype_number].cpu().numpy()

            # Resize to the input image dimensions
            activation_map = cv2.resize(
                activation_map,
                (experiment.img_size, experiment.img_size)
            )

            # Normalize for visualization
            activation_min = activation_map.min()
            activation_max = activation_map.max()

            if activation_max > activation_min:
                activation_map = (
                    (activation_map - activation_min)
                    / (activation_max - activation_min)
                )
            else:
                activation_map = np.zeros_like(activation_map)

            # Save the ProtoPNet activation heatmap
            activation_path = prototype_output_dir / f"ppnet_activation_{test_image_path.stem}.png"

            plt.imsave(
                activation_path,
                activation_map,
                cmap="jet",
                vmin=0,
                vmax=1
            )

            # Overlay the normalized ProtoPNet activation on the input image.
            activation_colormap = plt.get_cmap("jet")(activation_map)[..., :3]
            activation_overlay = (
                0.6 * original + 0.4 * activation_colormap
            ).clip(0, 1)
            activation_overlay_path = (
                prototype_output_dir
                / f"ppnet_activation_overlay_{test_image_path.stem}.png"
            )
            plt.imsave(activation_overlay_path, activation_overlay)

    print(f"Finished experiment: {experiment.model}")
    print(f"Results saved to: {output_dir}")
