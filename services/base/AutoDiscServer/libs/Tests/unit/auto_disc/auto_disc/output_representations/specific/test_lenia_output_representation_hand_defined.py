from auto_disc.output_representations.specific.lenia_output_representation_hand_defined import calc_image_moments, calc_distance_matrix, center_of_mass
from auto_disc.output_representations.specific import LeniaHandDefinedRepresentation
import os
import pickle
import sys
import unittest

import torch

classToTestFolderPath = os.path.dirname(__file__)
auto_discFolderPath = os.path.abspath(os.path.join(
    classToTestFolderPath, "../"*7 + "/libs/auto_disc/auto_disc"))
sys.path.insert(0, os.path.dirname(auto_discFolderPath))

__location__ = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))

Object = lambda **kwargs: type("Object", (), kwargs)()

# region calc_static_statistics


def test_calc_static_statistics_1():
    final_obs = torch.zeros(256, 256)
    leniaHandDefinedRepresentation = LeniaHandDefinedRepresentation()
    res = leniaHandDefinedRepresentation.calc_static_statistics(final_obs)
    resWeWant = torch.tensor(
        [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        dtype=torch.float64
    )
    assert torch.equal(res, resWeWant)


def test_calc_static_statistics_2():
    final_obs = torch.ones(256, 256)
    leniaHandDefinedRepresentation = LeniaHandDefinedRepresentation()
    res = leniaHandDefinedRepresentation.calc_static_statistics(final_obs)
    resWeWant = torch.tensor(
        [
            1.0, 1.0, 1.0, 0.25147351488655, 0.16666412353515625,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03888719348469749,
            0.0, -0.0, 0.0, -0.0
        ],
        dtype=torch.float64
    )
    assert torch.equal(res, resWeWant)


def test_calc_static_statistics_3():
    torch.manual_seed(0)
    final_obs = torch.rand(256, 256)
    leniaHandDefinedRepresentation = LeniaHandDefinedRepresentation()
    res = leniaHandDefinedRepresentation.calc_static_statistics(final_obs)
    resWeWant = torch.tensor(
        [
            0.5003275197811999, 0.999908447265625, 0.5003733303277997, 0.251042452716869, 0.3336547799369659, 7.096690293873195e-07,
            2.1693111783614055e-06, 1.892145736541528e-07, -
            5.294200699107513e-14, 8.042467397417452e-11, 1.0905359128915116e-13,
            -6.881052303613885e-11, 0.15577746476012266, 4.769783318058442e-11, 1.3297449584724628e-10, -
            2.270124137981597e-15, 7.771798476257806e-16
        ],
        dtype=torch.float64
    )
    assert torch.equal(res, resWeWant)


def test_calc_static_statistics_4():
    torch.manual_seed(1)
    final_obs = torch.rand(256, 256)
    leniaHandDefinedRepresentation = LeniaHandDefinedRepresentation()
    res = leniaHandDefinedRepresentation.calc_static_statistics(final_obs)
    resWeWant = torch.tensor(
        [
            0.5013266612175753, 0.999969482421875, 0.5013419609600362, 0.25140629478866405, 0.33238489270264604,
            4.6360606804069515e-07, 7.131791945215514e-08, 5.432964490055314e-08, -
            2.9254812163413216e-15, -7.129798513879064e-12,
            1.6966096858781673e-15, -1.814935802096843e-11, 0.1545789398774088, -
            1.664803567651011e-11, 1.5720882424051886e-11,
            -9.202165671862218e-17, 1.7226814748088533e-16
        ],
        dtype=torch.float64
    )
    assert torch.equal(res, resWeWant)

# endregion

# region calc_distance


def test_calc_distance_1():
    leniaOutputRepresentation = LeniaHandDefinedRepresentation("wrapped_key")
    embedding_a = torch.ones(2, 10)
    embedding_b = torch.zeros(2, 10)
    res = leniaOutputRepresentation.calc_distance(embedding_a, embedding_b)
    resWeWant = torch.tensor(
        [3.1622776601683795, 3.1622776601683795], dtype=torch.float64)
    assert torch.equal(res, resWeWant)


def test_calc_distance_2():
    leniaOutputRepresentation = LeniaHandDefinedRepresentation("wrapped_key")
    torch.manual_seed(0)
    embedding_a = torch.rand(2, 10)
    embedding_b = torch.rand(2, 10)
    res = leniaOutputRepresentation.calc_distance(embedding_a, embedding_b)
    resWeWant = torch.tensor(
        [1.1811040564038082, 1.048871181957222], dtype=torch.float64)
    assert torch.equal(res, resWeWant)


def test_calc_distance_3():
    leniaOutputRepresentation = LeniaHandDefinedRepresentation("wrapped_key")
    torch.manual_seed(1)
    embedding_a = torch.rand(2, 10)
    embedding_b = torch.rand(2, 10)
    res = leniaOutputRepresentation.calc_distance(embedding_a, embedding_b)
    resWeWant = torch.tensor(
        [1.6043511592139514, 1.1691390519147211], dtype=torch.float64)
    assert torch.equal(res, resWeWant)


def test_calc_distance_4():
    leniaOutputRepresentation = LeniaHandDefinedRepresentation("wrapped_key")
    torch.manual_seed(1)
    embedding_a = torch.rand(2, 10)
    embedding_b = torch.rand(2, 10)
    leniaOutputRepresentation.config.distance_function = "NotImplementedFunction"
    with unittest.TestCase.assertRaises(Exception, NotImplementedError) as context:
        res = leniaOutputRepresentation.calc_distance(embedding_a, embedding_b)
    assert type(context.exception).__name__ == 'NotImplementedError'

# endregion

# region calc_image_moments


def test_calc_image_moments_1():
    torch.manual_seed(0)
    image = torch.rand(256, 256)
    res = calc_image_moments(image)
    resWeWant = [
        127.5884228275033, 127.21352061409877, 32789.46433638072, 4183556.040037483,
        4171263.197281425, 531782138.8752178, 713299983.6007779, 709843003.5664766,
        90565109273.62798, 90424233655.4742, 15388495359666.68, 136771415996.33768, 17345154448572.457,
        17310120663030.262, 135970718364.66576, 27966246364261.742, 27787477584425.258, -
        422753.6643279195,
        179526666.6419202, 179201926.8322845, -
        48452418.81262207, 75274771.35791016, -68415947.65357971,
        -35955660.727005005, 981797518840.5107, -
        7185876919.93401, -3922082735.6446877, 1766903872327.5967,
        1761211112382.999, -0.0003932047330624085, 0.16697841081222364, 0.16667636912474226, -
        0.00024887407436987306,
        0.00038664610568881337, -0.00035141600897717703, -
        0.0001846849342320029, 0.027849600173562687,
        -0.00020383408521233354, -
        0.00011125352625084527, 0.05011987242293775, 0.04995839199005954, 0.3336547799369659,
        7.096690293873195e-07, 2.1693111783614055e-06, 1.892145736541528e-07, -
        5.294200699107513e-14, 8.042467397417452e-11,
        1.0905359128915116e-13, -
        6.881052303613885e-11, 0.15577746476012266, 4.769783318058442e-11, 1.3297449584724628e-10,
        -2.270124137981597e-15, 7.771798476257806e-16
    ]
    assert [x.item() for x in list(res.values())] == resWeWant

# endregion

# region calc_distance_matrix


def test_calc_distance_matrix_1():
    res = calc_distance_matrix(256, 256)
    with open(os.path.join(__location__+"/data", "test_calc_distance_matrix_1.pickle"), "rb") as resFile:
        saved_res = pickle.load(resFile)
    assert res.tolist() == saved_res

# endregion

# region calc_distance_matrix


def test_center_of_mass_1():
    res = center_of_mass(torch.zeros(256, 256))
    assert res.tolist() == [127, 127]

# endregion
