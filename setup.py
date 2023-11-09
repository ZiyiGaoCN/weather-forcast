from setuptools import setup, find_packages

setup(
    name='weather-forcast',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g.
        # 'requests>=2.25.1',
    ],
    description='A simple weather forecasting package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)

