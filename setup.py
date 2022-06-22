from setuptools import setup, find_packages


setup(name='reid',
      version='0.1.0',
      description='re-identification',
      author='dpjin',
      url='https://github.com/dapeng/reid',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu==1.6.3'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Object Re-identification'
      ])
