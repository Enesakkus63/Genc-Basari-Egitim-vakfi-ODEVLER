from araba import Sedan, SUV, SportsCar
from araba_galeri import CarGallery
gallery = CarGallery("Hızlı ve Güzel Arabalar")
sedan = Sedan("Toyota", "Corolla", 2022, 500000)
suv = SUV("Ford", "Explorer", 2021, 800000)
sports_car = SportsCar("Ferrari", "488 GTB", 2023, 5000000)
gallery.add_car(sedan)
gallery.add_car(suv)
gallery.add_car(sports_car)
gallery.list_cars()
