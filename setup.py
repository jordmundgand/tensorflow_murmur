from setuptools import setup, find_packages


def readme():
  with open('README.md', 'r') as f:
    return f.read()


setup(
  name='tensorflow-murmur',
  version='0.0.0',
  author='Ivan V. Savkin',
  author_email='i.v.savkin2020@yandex.ru',
  description='This is small addons module for tensorflow',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='your_url',
  packages=find_packages(),
  install_requires=['tensorflow'],
  classifiers=[
    'Programming Language :: Python :: 3.11',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  keywords='tf addons',
  project_urls={
    'GitHub': 'your_github'
  },
  python_requires='>=3.6'
)
