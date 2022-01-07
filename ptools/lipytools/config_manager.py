"""
    Configuration Manager
        - keeps configuration {key: value}
        - loads from and saves to file
"""
from copy import deepcopy
from typing import Optional

from ptools.lipytools.little_methods import w_json, r_json


class ConfigManager:

    def __init__(
            self,
            config: Optional[dict], # {param: value}
            file: str,
            try_to_load=    True):  # tries to load from file if file exists

        self.__file = file
        self.__config = config if config is not None else {}
        if try_to_load:
            file_config = r_json(self.__file)
            if file_config is not None: self.__config = file_config
        self.__save_file()

    # loads configuration from file and returns it
    def __read_file(self) -> dict:
        return r_json(self.__file)

    # saves configuration to jsonl file
    def __save_file(self):
        w_json(self.__config, self.__file)

    def get_config(self) -> dict:
        return deepcopy(self.__config)

    # loads configuration from file, updates self, returns new configuration (from file) or only keys that have changed values
    def load(
            self,
            return_only_changed=    True) -> dict:

        file_config = self.__read_file()
        config_changed = {}
        for k in file_config:
            if k not in self.__config or self.__config[k] != file_config[k]:
                self.__config[k] = file_config[k]
                config_changed[k] = file_config[k]

        return config_changed if return_only_changed else file_config

    # updates self with given kwargs, saves file if needed
    def update(
            self,
            return_only_changed=    True,
            **kwargs) -> dict:

        config_changed = {}
        for k in kwargs:
            if k not in self.__config:
                self.__config[k] = kwargs[k]
                config_changed[k] = kwargs[k]
            else:
                if kwargs[k] != self.__config[k]:
                    self.__config[k] = kwargs[k]
                    config_changed[k] = kwargs[k]

        if config_changed: self.__save_file()

        return config_changed if return_only_changed else self.get_config()


def test_config(n_loops=10):

    import time

    config = {
        'param_aaa':    15,
        'beta':         20.45,
        'do_it':        False,
        'dont_do':      None}

    cm = ConfigManager(
        config= config,
        file=   'config.file')
    print(cm.get_config())

    for _ in range(n_loops):
        time.sleep(5)
        newc = cm.load()
        print(newc, cm.get_config())


if __name__ == '__main__':
    test_config()
