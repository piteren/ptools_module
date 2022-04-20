from setuptools import find_packages

# reads requirements from 'ptools/requirements.txt'
def get_requirements():
    with open('ptools/requirements.txt', 'r') as file:
        lines = [file.readline()[:-1] for line in file]
        return lines


if __name__ == '__main__':
    print(f'packages:     {find_packages()}')
    print(f'requirements: {get_requirements()}')