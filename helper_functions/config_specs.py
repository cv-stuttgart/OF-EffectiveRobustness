
with open('/etc/hostname') as f:
  hostname = f.read().split('.')[0].strip()

  if hostname in ['haendel', 'vangogh', 'lounge', 'sarinagara']:
    # For seemlessly switching between computers with shared home...
    print(f'host name "{hostname}"')

    if hostname == 'haendel':
      dataset_folder = '../symlinks/datasets_haendel'
    elif hostname == 'vangogh':
      dataset_folder = '../symlinks/datasets_vangogh'
    elif hostname == 'lounge':
      dataset_folder = '../symlinks/datasets_lounge'
    elif hostname == 'sarinagara':
      dataset_folder = '../symlinks/datasets_sarinagara'

  else:
    # default
    dataset_folder = './datasets'

class Paths:
  __conf = {
    # Insert paths/to/local/datasets here
    "sintel_mpi": f"{dataset_folder}/Sintel",
    "sintel_subsplit": f"{dataset_folder}/sintel_splitmsk",
    "flying_chairs": f"{dataset_folder}/FlyingChairs_release",
    "flying_things": f"{dataset_folder}/FlyingThings3D",
    "kitti15": f"{dataset_folder}/KITTI",
    "hd1k": f"{dataset_folder}/HD1k",
    "spring": f"{dataset_folder}/Spring",
    "driving": f"{dataset_folder}/driving",
    "viper": f"{dataset_folder}/Viper",
  }

  __splits = {
    # Used for dataloading internally
    "sintel_train": "training",
    "sintel_eval": "test",
    "sintel_sub_train": "sintel_train",
    "sintel_sub_eval": "sintel_eval",
    "kitti_train": "training",
    "kitti_eval": "testing",
    "spring_train": "train",
    "spring_eval": "test",
    "viper_train": "train",
    "viper_eval": "val",
  }

  @staticmethod
  def config(name):
    return Paths.__conf[name]

  @staticmethod
  def splits(name):
    return Paths.__splits[name]

class Conf:
  __conf = {
    # Change the following variables according to your system setup.
    "useCPU": False,  # affects all .to(device) calls

    # Set to False, if your installation of spatial-correlation-sampler
    "correlationSamplerOnlyCPU": True,  # only used for PWCNet

    "hostname": hostname
  }

  @staticmethod
  def config(name):
    return Conf.__conf[name]


class ProgBar:
  __settings= {
      "disable": False,
      "format_eval": "{desc:19}:{percentage:3.0f}%|{bar:40}{r_bar}",
      "format_train": "{desc:13}:{percentage:3.0f}%|{bar:40}{r_bar}"
  }

  @staticmethod
  def settings(name):
    return ProgBar.__settings[name]
