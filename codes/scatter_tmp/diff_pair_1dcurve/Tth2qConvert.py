import math


class Tth2qConvert:
    def __init__(self, xray_energy):
        self.XrayEnergy = xray_energy  # keV unit
        XrayWavelength = 12.3984 / self.XrayEnergy  # Angstrom unit (10^-10 m)
        self.QCoefficient = 4 * math.pi / XrayWavelength

    def tth_to_q(self, tth_arr):
        """
        convert 2theta value to q
        :param tth_arr: 2theta(degree unit) array
        :return: q value array
        """
        output_q = []
        for each_tth in tth_arr:
            now_q = self.QCoefficient * math.sin(math.radians(each_tth / 2))
            output_q.append(now_q)
        return output_q
