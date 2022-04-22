from setuptools import find_packages

# reads requirements from 'ptools/requirements.txt'
def get_requirements():
    with open('ptools/requirements.txt') as file:
        lines = [l[:-1] for l in file.readlines()]
        return lines


if __name__ == '__main__':
    print(f'packages:     {find_packages()}')
    print(f'requirements: {get_requirements()}')