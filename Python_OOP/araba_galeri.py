from araba import Car, Sedan, SUV, SportsCar
class CarGallery:
    def __init__(self, name):
        self.name = name
        self.cars = []
    def add_car(self, car: Car):
        self.cars.append(car)
        print(f"{car.brand} {car.get_model()} isimli araba galeriye eklendi.")
    def list_cars(self):
        print(f"{self.name} Galerisi'ndeki Arabalar:")
        for car in self.cars:
            print(f"- {car.brand} {car.get_model()} ({car.get_year()}), Fiyat: {car.get_price()} TL, Bilgi: {car.display_info()}")
