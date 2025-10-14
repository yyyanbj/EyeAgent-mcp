import torch.utils.data as data
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms as T
import pandas as pd
import numpy as np
from PIL import Image, ImageFile

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks")

ImageFile.LOAD_TRUNCATED_IMAGES = True

from medical_diffusion.data.augmentation.augmentations_2d import Normalize, ToTensor16bit


class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers=[],
        crawler_ext="tif",  # other options are ['jpg', 'jpeg', 'png', 'tiff'],
        transform=None,
        image_resize=None,
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        image_crop=None,
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext)

        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize(image_resize) if image_resize is not None else nn.Identity(),
                    T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                    T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                    # T.CenterCrop(image_crop) if image_crop is not None else nn.Identity(),
                    T.RandomCrop(image_crop) if image_crop is not None else nn.Identity(),
                    T.ToTensor(),
                    # T.Lambda(lambda x: torch.cat([x]*3) if x.shape[0]==1 else x),
                    # ToTensor16bit(),
                    # Normalize(), # [0, 1.0]
                    # T.ConvertImageDtype(torch.float),
                    T.Normalize(
                        mean=0.5, std=0.5
                    ),  # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        # img = Image.open(path_item)
        img = self.load_item(path_item)
        return {"uid": rel_path_item.stem, "source": self.transform(img)}

    def load_item(self, path_item):
        return Image.open(path_item).convert("RGB")
        # return cv2.imread(str(path_item), cv2.IMREAD_UNCHANGED) # NOTE: Only CV2 supports 16bit RGB images

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f"*.{extension}")]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None


class AIROGSDataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = pd.read_csv(
            self.path_root.parent / "train_labels.csv", index_col="challenge_id"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        uid = self.labels.index[index]
        path_item = self.path_root / f"{uid}.jpg"
        img = self.load_item(path_item)
        str_2_int = {"NRG": 0, "RG": 1}  # RG = 3270, NRG = 98172
        target = str_2_int[self.labels.loc[uid, "class"]]
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {"source": self.transform(img), "target": target}

    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1 / self.labels["class"].value_counts(
            normalize=True
        )  # {'NRG': 1.03, 'RG': 31.02}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.iloc[index]["class"]
            weights[index] = weight_per_class[target]
        return weights

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []


class MSIvsMSS_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.2530835
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {"MSIMUT": 0, "MSS": 1}
        target = str_2_int[path_item.parent.name]  #
        return {"uid": uid, "source": self.transform(img), "target": target}


class MSIvsMSS_2_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.3832231
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {
            "MSIH": 0,
            "nonMSIH": 1,
        }  # patients with MSI-H = MSIH; patients with MSI-L and MSS = NonMSIH)
        target = str_2_int[path_item.parent.name]
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {"source": self.transform(img), "target": target}


class CheXpert_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = self.path_root.name
        labels = pd.read_csv(self.path_root.parent / f"{mode}.csv", index_col="Path")
        self.labels = labels.loc[labels["Frontal/Lateral"] == "Frontal"].copy()
        self.labels.index = self.labels.index.str[20:]
        self.labels.loc[self.labels["Sex"] == "Unknown", "Sex"] = (
            "Female"  # Affects 1 case, must be "female" to match stats in publication
        )
        self.labels.fillna(2, inplace=True)  # TODO: Find better solution,
        str_2_int = {
            "Sex": {"Male": 0, "Female": 1},
            "Frontal/Lateral": {"Frontal": 0, "Lateral": 1},
            "AP/PA": {"AP": 0, "PA": 1},
        }
        self.labels.replace(str_2_int, inplace=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        rel_path_item = self.labels.index[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        uid = str(rel_path_item)
        target = torch.tensor(
            self.labels.loc[uid, "Cardiomegaly"] + 1, dtype=torch.long
        )  # Note Labels are -1=uncertain, 0=negative, 1=positive, NA=not reported -> Map to [0, 2], NA=3
        return {"uid": uid, "source": self.transform(img), "target": target}

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []


class CheXpert_2_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = pd.read_csv(
            self.path_root / "labels/cheXPert_label.csv", index_col=["Path", "Image Index"]
        )  # Note: 1 and -1 (uncertain) cases count as positives (1), 0 and NA count as negatives (0)
        labels = labels.loc[labels["fold"] == "train"].copy()
        labels = labels.drop(labels="fold", axis=1)

        labels2 = pd.read_csv(self.path_root / "labels/train.csv", index_col="Path")
        labels2 = labels2.loc[labels2["Frontal/Lateral"] == "Frontal"].copy()
        labels2 = labels2[
            [
                "Cardiomegaly",
            ]
        ].copy()
        labels2[(labels2 < 0) | labels2.isna()] = 2  # 0 = Negative, 1 = Positive, 2 = Uncertain
        labels = labels.join(
            labels2["Cardiomegaly"],
            on=[
                "Path",
            ],
            rsuffix="_true",
        )
        # labels = labels[labels['Cardiomegaly_true']!=2]

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path_index, image_index = self.labels.index[index]
        path_item = self.path_root / "data" / f"{image_index:06}.png"
        img = self.load_item(path_item)
        uid = image_index
        target = int(self.labels.loc[(path_index, image_index), "Cardiomegaly"])
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {"source": self.transform(img), "target": target}

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1 / self.labels["Cardiomegaly"].value_counts(normalize=True)
        # weight_per_class = {2.0: 1.2, 1.0: 8.2, 0.0: 24.3}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.loc[self.labels.index[index], "Cardiomegaly"]
            weights[index] = weight_per_class[target]
        return weights


class ExternalEye_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = pd.read_csv(
            "/data/LateOrchestration/External-Eye-Dataset/diffusion_data/condation_data_processed/factor_10.csv"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        info = self.labels.iloc[index]
        path_index = info["file name"]
        img = self.load_item(path_index)

        return {
            "source": self.transform(img),
        }


class Toy_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = pd.DataFrame(
            {
                "idx": list(range(10000)),
            }
        )
        print("# dataset length", len(labels))
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def load_item(
        self,
    ):
        # image_data = torch.randint(0, 256, size=(8, 28, 28))
        # image_data = image_data / 255.
        image_data = torch.randint(0, 256, size=(3, 224, 224))
        image_data = image_data / 255.0
        return image_data

    def __getitem__(self, index):

        img = self.load_item()
        eye_target = torch.tensor(0, dtype=torch.long)
        biomarker_target = torch.tensor([-1], dtype=torch.float)
        age_target = torch.tensor([float(-1)], dtype=torch.float)
        gender_target = torch.tensor(0, dtype=torch.long)

        return {
            "source": self.transform(img),
            "eye": eye_target,
            "biomarker": biomarker_target,
            "age": age_target,
            "gender": gender_target,
        }

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []


class ExternalEye_Dataset_2(SimpleDataset2D):
    def __init__(self, task_id=34, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = pd.read_csv(
            "/data/LateOrchestration/External-Eye-Dataset/diffusion_data/condation_data_processed/factor_{}.csv".format(
                task_id
            )
        )
        # labels = labels.dropna(subset=["age", "gender"]).reset_index(drop=True)
        print("# dataset length", len(labels))
        self.labels = labels
        # self.task_id = task_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        info = self.labels.iloc[index]
        path_index = info["file name"]
        img = self.load_item(path_index)

        # EYE
        eye_info = info["eye"]
        if pd.isna(eye_info):
            eye_target = torch.tensor(0, dtype=torch.long)
        else:
            eye_target = torch.tensor(["L", "R"].index(eye_info) + 1, dtype=torch.long)

        # BIOMARKER
        biomarker_info = info["category"]
        if pd.isna(biomarker_info):
            biomarker_target = torch.tensor([-1], dtype=torch.float)
        else:
            biomarker_target = torch.tensor([float(biomarker_info)], dtype=torch.float)

        # ATTRIBUTES
        age_info = info["age"]
        if pd.isna(age_info):
            age_target = torch.tensor([float(-1)], dtype=torch.float)
        elif float(age_info) < 30 or float(age_info) > 80:
            age_target = torch.tensor([float(-1)], dtype=torch.float)
        else:
            age_target = torch.tensor([float(age_info)], dtype=torch.float)

        gender_info = info["gender"]
        if pd.isna(gender_info):
            gender_target = torch.tensor(0, dtype=torch.long)
        else:
            gender_target = torch.tensor(
                ["Female", "Male"].index(gender_info) + 1, dtype=torch.long
            )

        return {
            "source": self.transform(img),
            "eye": eye_target,
            "biomarker": biomarker_target,
            "age": age_target,
            "gender": gender_target,
        }

        # return {'source': self.transform(img),
        #         # "target": biomarker_target,
        #         'disease': disease_target,
        #         'biomarker': biomarker_target,
        #         'age': age_target,
        #         "gender": gender_target}

        # return {'source': self.transform(img),
        #         "target": biomarker_target,}

        # category = info["category"]
        # target = ("Low", "High").index(category)
        # # return {'uid':uid, 'source': self.transform(img), 'target':target}
        # return {'source': self.transform(img), 'target':target}

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

    # def get_weights(self):
    #     n_samples = len(self)
    #     weight_per_class = 1/self.labels["hospital ID"].value_counts(normalize=True)
    #     print(weight_per_class)
    #     # weight_per_class = {2.0: 1.2, 1.0: 8.2, 0.0: 24.3}
    #     # weight_per_class = {"High": 0.8, False: 0.2}
    #     weights = [0] * n_samples
    #     for index in range(n_samples):
    #         # target = self.labels.loc[self.labels.index[index], 'Cardiomegaly']
    #         category = self.labels.loc[index, "hospital ID"]
    #         # target = ("Low", "High").index(category)
    #         weights[index] = weight_per_class[category]
    #     return weights

    # def get_disease_labels(self, d_list):
    #     d_tensor = torch.zeros((9))
    #     for d in d_list:
    #         d_tensor[int(d)] = 1.
    #     return d_tensor


class MeshFusion_CFP_Dataset(SimpleDataset2D):
    def __init__(self, num_ref_vertex, training, subset_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if training:
            labels = pd.read_csv(
                "dataset/meshfusion_cfp_pts{}/train_{}.csv".format(num_ref_vertex, subset_idx)
            )
        else:
            labels = pd.read_csv(
                "dataset/meshfusion_cfp_pts{}/val_{}.csv".format(num_ref_vertex, subset_idx)
            )
        print("# dataset length", len(labels))
        self.labels = labels

        self.mesh_dir = "ocular_latent/{}pts/".format(num_ref_vertex)
        self.cfp_dir = "latent_fundus/"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        info = self.labels.iloc[index]

        mesh_array = np.load(self.mesh_dir + info["alpha"])  # (111, )
        # if self.reshape_idx == 1:
        mesh_latent = torch.tensor(mesh_array, dtype=torch.float).view(-1, 1, 1)
        mesh_latent = mesh_latent.repeat(1, 28, 28)
        assert mesh_latent.shape[0] == 113
        # else:
        #     mesh_latent = torch.tensor(mesh_array, dtype=torch.float)
        #     mesh_latent = F.pad(mesh_latent, (0, int(self.spatial_dim[self.embed_dim] - self.embed_dim))).view(1, -1, 1)  # (128, )
        #     mesh_latent = mesh_latent.repeat(8, 1, self.spatial_dim[self.embed_dim])

        # EYE
        eye_info = info["eye"]
        eye_target = torch.tensor(["OS", "OD"].index(eye_info), dtype=torch.long)

        # Fundus
        fundus_latent = np.load(self.cfp_dir + info["fundus"] + ".npy")
        fundus_latent = torch.tensor(fundus_latent, dtype=torch.float)

        return {
            "source": mesh_latent,
            "eye": eye_target,
            "fundus": fundus_latent,
        }

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []


def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma**2))) / (np.sqrt(2 * np.pi) * sigma)


def add_meta_info(src_df, com_df):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    for i in range(len(src_df)):
        info = src_df.iloc[i]

        # query info
        idx_info = info["idx"]
        eye_info = info["eye"]

        # find
        indices = com_df.index[(com_df["idx"] == idx_info) & (com_df["EYE"] == eye_info)].tolist()
        if len(indices) == 1:
            tar_idx = indices[0]
            list1.append(com_df.loc[tar_idx, "age"])
            list2.append(com_df.loc[tar_idx, "gender"])
            list3.append(com_df.loc[tar_idx, "sph"])
            list4.append(com_df.loc[tar_idx, "al_"])

        else:
            list1.append(np.NaN)
            list2.append(np.NaN)
            list3.append(np.NaN)
            list4.append(np.NaN)

    src_df["age"] = list1
    src_df["gender"] = list2
    src_df["sph"] = list3
    src_df["al"] = list4
    src_df["has_nan"] = src_df.isna().any(axis=1)
    src_df = src_df[src_df["has_nan"] == False].reset_index(drop=True)
    return src_df


class MeshFusion_CFP_META_Dataset(SimpleDataset2D):
    def __init__(self, num_ref_vertex, training, subset_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if training:
            labels = pd.read_csv(
                "dataset/meshfusion_cfp_pts{}/train_{}.csv".format(num_ref_vertex, subset_idx)
            )
        else:
            labels = pd.read_csv(
                "dataset/meshfusion_cfp_pts{}/val_{}.csv".format(num_ref_vertex, subset_idx)
            )
        print("# dataset length", len(labels))

        meta = pd.read_csv("dataset/metadata.csv")
        labels_added = add_meta_info(labels, meta)
        print("# actual dataset length", len(labels_added))

        self.labels = labels_added
        self.mesh_dir = "ocular_latent/{}pts/".format(num_ref_vertex)
        self.cfp_dir = "latent_fundus/"
        # self.cfp_dir = "latent_fundus_eyefound/"

        # meta encode
        self.ld_sigma = 1.0
        # num_elements = int(self.meta["sph"].max() - self.meta["sph"].min() + 2)
        num_elements = 512
        self.x_sph = np.linspace(
            self.labels["sph"].min() - self.ld_sigma,
            self.labels["sph"].max() + self.ld_sigma,
            num_elements,
        )
        self.x_al = np.linspace(
            self.labels["al"].min() - self.ld_sigma,
            self.labels["al"].max() + self.ld_sigma,
            num_elements,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        info = self.labels.iloc[index]

        mesh_array = np.load(self.mesh_dir + info["alpha"])  # (111, )
        # if self.reshape_idx == 1:
        mesh_latent = torch.tensor(mesh_array, dtype=torch.float).view(-1, 1, 1)
        mesh_latent = mesh_latent.repeat(1, 28, 28)
        assert mesh_latent.shape[0] == 113

        # EYE
        eye_info = info["eye"]
        eye_target = torch.tensor(["OS", "OD"].index(eye_info), dtype=torch.long)

        # Fundus
        fundus_latent = np.load(self.cfp_dir + info["fundus"] + ".npy")
        fundus_latent = torch.tensor(fundus_latent, dtype=torch.float)

        # Meta Info
        age_info = info["age"]
        gender_info = info["gender"]
        sph_info = info["sph"]
        al_info = info["al"]

        age_target = (
            torch.tensor([round(age_info)], dtype=torch.float) if not np.isnan(age_info) else None
        )
        gender_target = (
            torch.tensor([int(gender_info - 1)], dtype=torch.float)
            if not np.isnan(gender_info)
            else None
        )
        # sph_target = torch.tensor([float(sph_info)], dtype=torch.float) if not np.isnan(sph_info) else None
        # al_target = torch.tensor([float(al_info)], dtype=torch.float) if not np.isnan(al_info) else None

        if not np.isnan(sph_info):
            sph_array = self.label_distribution_encode(self.x_sph, sph_info, self.ld_sigma)
            sph_target = torch.tensor(sph_array, dtype=torch.float)
        else:
            sph_target = None

        if not np.isnan(al_info):
            al_array = self.label_distribution_encode(self.x_al, al_info, self.ld_sigma)
            al_target = torch.tensor(al_array, dtype=torch.float)
        else:
            al_target = None

        return {
            "source": mesh_latent,
            "eye": eye_target,
            "fundus": fundus_latent,
            "age": age_target,
            "gender": gender_target,
            "sph": sph_target,
            "al": al_target,
        }

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

    def label_distribution_encode(self, x_idx, mu_, sigma_):
        y_ = normal_distribution(x_idx, mu_, sigma_)
        return y_


class MeshFusion_CFP_META_MLP_Dataset(SimpleDataset2D):
    def __init__(self, num_ref_vertex, training, subset_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if training:
            labels = pd.read_csv(
                "dataset/meshfusion_cfp_pts{}/train_{}.csv".format(num_ref_vertex, subset_idx)
            )
        else:
            labels = pd.read_csv(
                "dataset/meshfusion_cfp_pts{}/val_{}.csv".format(num_ref_vertex, subset_idx)
            )
        print("# dataset length", len(labels))

        meta = pd.read_csv("dataset/metadata.csv")
        labels_added = add_meta_info(labels, meta)
        print("# actual dataset length", len(labels_added))

        self.labels = labels_added
        self.mesh_dir = "ocular_latent/{}pts/".format(num_ref_vertex)
        self.cfp_dir = "latent_fundus/"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        info = self.labels.iloc[index]

        mesh_array = np.load(self.mesh_dir + info["alpha"])  # (111, )
        # if self.reshape_idx == 1:
        mesh_latent = torch.tensor(mesh_array, dtype=torch.float).view(-1, 1, 1)
        mesh_latent = mesh_latent.repeat(1, 28, 28)
        assert mesh_latent.shape[0] == 113

        # EYE
        eye_info = info["eye"]
        eye_target = torch.tensor(["OS", "OD"].index(eye_info), dtype=torch.long)

        # Fundus
        fundus_latent = np.load(self.cfp_dir + info["fundus"] + ".npy")
        fundus_latent = torch.tensor(fundus_latent, dtype=torch.float)

        # Meta Info
        age_info = info["age"]
        gender_info = info["gender"]
        sph_info = info["sph"]
        al_info = info["al"]

        age_target = (
            torch.tensor([round(age_info)], dtype=torch.float) if not np.isnan(age_info) else None
        )
        gender_target = (
            torch.tensor([int(gender_info - 1)], dtype=torch.float)
            if not np.isnan(gender_info)
            else None
        )
        sph_target = (
            torch.tensor([float(sph_info)], dtype=torch.float) if not np.isnan(sph_info) else None
        )
        al_target = (
            torch.tensor([float(al_info)], dtype=torch.float) if not np.isnan(al_info) else None
        )

        return {
            "source": mesh_latent,
            "eye": eye_target,
            "fundus": fundus_latent,
            "age": age_target,
            "gender": gender_target,
            "sph": sph_target,
            "al": al_target,
        }

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []


class MeshFusion_CFP_META_KMeans_Dataset(SimpleDataset2D):
    def __init__(self, num_ref_vertex, training, subset_idx, num_clusters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if training:
            labels = pd.read_csv(
                "dataset/meshfusion_cfp_pts{}/train_{}.csv".format(num_ref_vertex, subset_idx)
            )
        else:
            labels = pd.read_csv(
                "dataset/meshfusion_cfp_pts{}/val_{}.csv".format(num_ref_vertex, subset_idx)
            )
        print("# dataset length", len(labels))

        meta = pd.read_csv("dataset/metadata.csv")
        labels_added = add_meta_info(labels, meta)
        print("# actual dataset length", len(labels_added))

        self.labels = labels_added
        self.mesh_dir = "ocular_latent/{}pts/".format(num_ref_vertex)
        self.cfp_dir = "latent_fundus/"
        self.num_ref_vertex, self.subset_idx = num_ref_vertex, subset_idx
        self.num_clusters = num_clusters
        self.feature_list = ["sph", "al"]

        if training:
            self.get_kmeans_label_train_phase()
        else:
            self.get_kmeans_label_val_phase()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        info = self.labels.iloc[index]

        mesh_array = np.load(self.mesh_dir + info["alpha"])  # (111, )
        # if self.reshape_idx == 1:
        mesh_latent = torch.tensor(mesh_array, dtype=torch.float).view(-1, 1, 1)
        mesh_latent = mesh_latent.repeat(1, 28, 28)
        assert mesh_latent.shape[0] == 113

        # EYE
        eye_info = info["eye"]
        eye_target = torch.tensor(["OS", "OD"].index(eye_info), dtype=torch.long)

        # Fundus
        fundus_latent = np.load(self.cfp_dir + info["fundus"] + ".npy")
        fundus_latent = torch.tensor(fundus_latent, dtype=torch.float)

        # Meta Info
        age_info = info["age"]
        gender_info = info["gender"]
        sph_info = info["sph"]
        al_info = info["al"]
        meta_cluster_info = info["meta_cluster"]

        age_target = (
            torch.tensor([round(age_info)], dtype=torch.float) if not np.isnan(age_info) else None
        )
        gender_target = (
            torch.tensor([int(gender_info - 1)], dtype=torch.float)
            if not np.isnan(gender_info)
            else None
        )
        sph_target = (
            torch.tensor([float(sph_info)], dtype=torch.float) if not np.isnan(sph_info) else None
        )
        al_target = (
            torch.tensor([float(al_info)], dtype=torch.float) if not np.isnan(al_info) else None
        )

        meta_target = torch.tensor(int(meta_cluster_info), dtype=torch.long)

        return {
            "source": mesh_latent,
            "eye": eye_target,
            "fundus": fundus_latent,
            "age": age_target,
            "gender": gender_target,
            "sph": sph_target,
            # 'al': al_target,
            "al": meta_target,
        }

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

    def get_kmeans_label_train_phase(
        self,
    ):
        df = self.labels.copy()
        feature_list = self.feature_list

        # get features
        data = []
        for f in feature_list:
            data.append(df[f].to_list())
        X_data = np.array(data).transpose(1, 0)
        assert X_data.shape[1] == len(feature_list)

        # clustering
        kmeans = KMeans(init="k-means++", n_clusters=self.num_clusters, n_init=2, random_state=0)
        y_data = kmeans.fit_predict(X_data)

        # re-assign based AL
        cluster_centers = kmeans.cluster_centers_
        cluster_centers_sort_info = cluster_centers[:, 1].argsort().tolist()

        sorted_cluster_centers = cluster_centers[cluster_centers[:, 1].argsort()]
        y_data = [cluster_centers_sort_info.index(element) for element in y_data]
        # y_data = np.array(y_data)

        self.labels["meta_cluster"] = y_data
        np.save(
            f"dataset/meshfusion_cfp_pts{self.num_ref_vertex}/kmeans{self.num_clusters}_centers_{self.subset_idx}.npy",
            sorted_cluster_centers,
        )

        self.labels.to_csv(
            f"dataset/meshfusion_cfp_pts{self.num_ref_vertex}/cluster{self.num_clusters}_train_{self.subset_idx}.csv"
        )

        sns.scatterplot(
            data=self.labels,
            x=feature_list[0],
            y=feature_list[1],
            hue="meta_cluster",
            palette="Paired",
        )
        plt.savefig(
            f"dataset/meshfusion_cfp_pts{self.num_ref_vertex}/cluster{self.num_clusters}_train_{self.subset_idx}.png",
            dpi=600,
        )
        plt.close()

    def get_kmeans_label_val_phase(
        self,
    ):
        df = self.labels.copy()
        feature_list = self.feature_list

        # query
        cluster_centers = np.load(
            f"dataset/meshfusion_cfp_pts{self.num_ref_vertex}/kmeans{self.num_clusters}_centers_{self.subset_idx}.npy"
        )
        y_data = []
        for i in range(len(df)):
            query_vector = []
            for f in feature_list:
                query_vector.append(df.loc[i, f])
            query_vector = np.array(query_vector).reshape(1, -1)

            # 计算每一行与给定中心之间的距离
            distances = np.linalg.norm(cluster_centers - query_vector, axis=1)

            # 找到距离最小的中心的下标
            min_index = np.argmin(distances)

            # 获取距离最近的中心
            # nearest_row = cluster_centers[min_index]

            y_data.append(min_index)

        self.labels["meta_cluster"] = y_data

        self.labels.to_csv(
            f"dataset/meshfusion_cfp_pts{self.num_ref_vertex}/cluster{self.num_clusters}_val_{self.subset_idx}.csv"
        )

        sns.scatterplot(
            data=self.labels,
            x=feature_list[0],
            y=feature_list[1],
            hue="meta_cluster",
            palette="Paired",
        )
        plt.savefig(
            f"dataset/meshfusion_cfp_pts{self.num_ref_vertex}/cluster{self.num_clusters}_val_{self.subset_idx}.png",
            dpi=600,
        )
        plt.close()
