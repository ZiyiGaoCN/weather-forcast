from setuptools import setup, find_packages

setup(
    name='weather_forcast',
    version='0.1.0',
    packages=find_packages(include=['weather_forcast']),
    install_requires=[
        # Add your dependencies here, e.g.
        # 'requests>=2.25.1',
    ],
    description='A simple weather forecasting package',
    
)
