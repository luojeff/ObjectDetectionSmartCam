from googlenet import googlenet

g = googlenet("deploy.prototxt", "googlenet_finetune.caffemodel")

print(g.analyze("test_cars/car1.jpg", "test_cars/car1.jpg"))
print(g.analyze("test_cars/car1.jpg", "test_cars/car2.jpg"))
print(g.analyze("test_cars/car2.jpg", "test_cars/car3.jpg"))
print(g.analyze("test_cars/car1.jpg", "test_cars/car3.jpg"))
