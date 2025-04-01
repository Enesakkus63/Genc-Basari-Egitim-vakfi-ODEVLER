from abc import ABC, abstractmethod
class Car(ABC):
    def __init__(self, brand, model, year, price):
        self.brand = brand
        self._model = model
        self._year = year
        self._price = price

    @abstractmethod
    def display_info(self):
        pass

    def get_model(self):
        return self._model

    def get_year(self):
        return self._year

    def get_price(self):
        return self._price

    def __str__(self):
        return f"{self.brand} {self._model} ({self._year})"

class Sedan(Car):
    def display_info(self):
        return f"Bu bir Sedan arabadır: {self.brand} {self._model}, Yıl: {self._year}, Fiyat: {self._price} TL"

class SUV(Car):
    def display_info(self):
        return f"Bu bir SUV arabadır: {self.brand} {self._model}, Yıl: {self._year}, Fiyat: {self._price} TL"

class SportsCar(Car):
    def display_info(self):
        return f"Bu bir Spor Arabadır: {self.brand} {self._model}, Yıl: {self._year}, Fiyat: {self._price} TL"
