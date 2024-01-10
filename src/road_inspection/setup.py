from setuptools import find_packages, setup

package_name = 'road_inspection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kasm-user',
    maintainer_email='hopelawrence070@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pothole_detector = road_inspection.pothole_detector:main',
            'mover = road_inspection.mover:main',
            'pothole_count = road_inspection.pothole_count:main'
        ],
    },
)
