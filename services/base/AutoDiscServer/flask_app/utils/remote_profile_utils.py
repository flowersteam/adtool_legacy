import yaml
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
profiles_path = os.path.join(
    dir_path, "../../../../configs/remote_experiments/profiles")


def parse_profile(path):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def list_profiles():
    profiles = []
    for (dirpath, _, filenames) in os.walk(profiles_path):
        for filename in filenames:
            if filename.endswith(".yml"):
                full_filename = os.path.join(dirpath, filename)
                profile = parse_profile(full_filename)
                profiles.append((profile["name"], full_filename))
    return profiles
