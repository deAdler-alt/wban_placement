# ../model/channel_model.py

from __future__ import annotations

from dataclasses import dataclass
from math import log10
from typing import Optional, Dict, Union

from body_map import BodyMap, Point


@dataclass
class PathLossParams:
    """
    Parametry modelu strat ścieżki dla danego typu łącza (LOS / NLOS).

    Atrybuty:
        n: wykładnik strat ścieżki (path-loss exponent)
        pl0_dB: strata odniesienia w odległości d0 [dB]
        d0_m: odległość odniesienia [m]
        sigma_shadow_dB: odchylenie standardowe składnika cieniowania [dB]
    """
    n: float
    pl0_dB: float
    d0_m: float
    sigma_shadow_dB: float = 0.0


class ChannelModel:
    """
    Model kanału on-body inspirowany IEEE 802.15.6 CM3 (uproszczony, uśredniony).

    Odpowiada zapisowi z rozdziału 3.2 pracy:
        PL(d) = PL_0 + 10 n log10(d/d_0)

    Losowy składnik cieniowania S nie jest domyślnie dodawany, aby zachować
    deterministyczny charakter funkcji celu (istotne dla optymalizacji).
    """

    def __init__(
        self,
        body_map: Optional[BodyMap] = None,
        los_params: Optional[PathLossParams] = None,
        nlos_params: Optional[PathLossParams] = None,
        rx_sensitivity_dBm: float = -90.0,
        safety_margin_dB: float = 10.0,
    ) -> None:
        # jeżeli nie podano, twórz domyślną mapę ciała
        self.body_map = body_map or BodyMap()

        # Domyślne parametry inspirowane pomiarami on-body dla pasma 2.4 GHz.
        self.los_params = los_params or PathLossParams(
            n=2.2,      # połączenie wzdłuż kończyny / tej samej strony tułowia
            pl0_dB=45,  # strata odniesienia w d0
            d0_m=0.1,
            sigma_shadow_dB=3.0,
        )
        self.nlos_params = nlos_params or PathLossParams(
            n=4.5,      # połączenie przez tors / przeciwna strona ciała
            pl0_dB=45,
            d0_m=0.1,
            sigma_shadow_dB=7.0,
        )

        self.rx_sensitivity_dBm = rx_sensitivity_dBm
        self.safety_margin_dB = safety_margin_dB

    # ------------------------------------------------------------------
    # Podstawowe obliczenia strat i mocy nadawania
    # ------------------------------------------------------------------
    def classify_link(
        self,
        p_tx: Point,
        region_tx: str,
        p_rx: Point,
        region_rx: str,
    ) -> str:
        """
        Zwraca typ łącza ('LOS' lub 'NLOS') na podstawie modelu ciała.

        W praktyce deleguje do BodyMap.classify_link, ale metoda jest wydzielona
        dla czytelności oraz ewentualnych przyszłych modyfikacji.
        """
        if self.body_map is None:
            raise ValueError("ChannelModel: body_map is required to classify links.")
        return self.body_map.classify_link(p_tx, region_tx, p_rx, region_rx)

    def path_loss_dB(
        self,
        d_m: float,
        link_type: str = "LOS",
        include_shadow: bool = False,
        shadow_sample_dB: Optional[float] = None,
    ) -> float:
        """
        Oblicza średnie straty ścieżki PL(d) [dB] dla danego typu łącza.

        Args:
            d_m: odległość geometryczna [m]
            link_type: 'LOS' lub 'NLOS'
            include_shadow: jeżeli True, do wyniku dodawany jest składnik
                cieniowania (podany w shadow_sample_dB).
            shadow_sample_dB: opcjonalny zrealizowany próg cieniowania [dB].
                Funkcja sama nie generuje wartości losowych – możesz wylosować
                S~N(0,σ) osobno i przekazać tutaj.

        Returns:
            Straty ścieżki w dB.
        """
        if d_m <= 0:
            raise ValueError("Distance must be positive")

        params = self.los_params if link_type.upper() == "LOS" else self.nlos_params
        d_eff = max(d_m, params.d0_m)  # zabezpieczenie dla d < d0

        pl = params.pl0_dB + 10.0 * params.n * log10(d_eff / params.d0_m)

        if include_shadow and shadow_sample_dB is not None:
            pl += shadow_sample_dB

        return pl

    def required_tx_power_dBm(self, path_loss_dB: float) -> float:
        """
        Minimalna wymagana moc nadawania P_TX [dBm] dla danego path-loss.

        Zgodnie z równaniem:
            P_TX[dBm] = P_RX_min[dBm] + PL[dB] + M_safety[dB]
        """
        return self.rx_sensitivity_dBm + path_loss_dB + self.safety_margin_dB

    @staticmethod
    def dBm_to_mW(p_dBm: float) -> float:
        """Konwersja mocy z dBm na mW."""
        return 10 ** (p_dBm / 10.0)

    def tx_energy_J(
        self,
        path_loss_dB: float,
        packet_length_bits: int,
        bit_rate_bps: float,
    ) -> float:
        """
        Energia transmisji pakietu [J] wynikająca z wymaganej mocy nadajnika.

        Przybliżenie:
            T_TX = packet_length_bits / bit_rate_bps
            P_TX[mW] = 10^(P_TX[dBm] / 10)
            E_TX[J] = P_TX[W] * T_TX

        Z punktu widzenia optymalizacji istotne są przede wszystkim RELACJE
        między energiami dla różnych rozmieszczeń, dlatego pomijamy stałe
        składowe (np. energię elektroniki).
        """
        if packet_length_bits <= 0:
            raise ValueError("packet_length_bits must be positive")
        if bit_rate_bps <= 0:
            raise ValueError("bit_rate_bps must be positive")

        p_tx_dBm = self.required_tx_power_dBm(path_loss_dB)
        p_tx_mW = self.dBm_to_mW(p_tx_dBm)
        p_tx_W = p_tx_mW / 1000.0

        t_tx = packet_length_bits / float(bit_rate_bps)
        return p_tx_W * t_tx

    # ------------------------------------------------------------------
    # Funkcja pomocnicza: kompletne metryki łącza dla dwóch węzłów
    # ------------------------------------------------------------------
    def link_metrics(
        self,
        p_tx: Point,
        region_tx: str,
        p_rx: Point,
        region_rx: str,
        packet_length_bits: int,
        bit_rate_bps: float,
        include_shadow: bool = False,
        shadow_sample_dB: Optional[float] = None,
    ) -> Dict[str, Union[float, int]]:
        """
        Wygodna funkcja zwracająca komplet metryk łącza dla dwóch węzłów.

        Zwracane wartości:
            - 'distance_m'  : odległość geometryczna
            - 'link_type'   : typ łącza (1 = LOS, 0 = NLOS – kompatybilne z fitness)
            - 'pl_dB'       : straty ścieżki
            - 'p_tx_dBm'    : wymagana moc nadawania
            - 'energy_J'    : energia transmisji pakietu
        """
        link_type = self.classify_link(p_tx, region_tx, p_rx, region_rx)
        d_m = self.body_map.distance(p_tx, p_rx)

        pl_dB = self.path_loss_dB(
            d_m=d_m,
            link_type=link_type,
            include_shadow=include_shadow,
            shadow_sample_dB=shadow_sample_dB,
        )
        p_tx_dBm = self.required_tx_power_dBm(pl_dB)
        energy_J = self.tx_energy_J(pl_dB, packet_length_bits, bit_rate_bps)

        return {
            "distance_m": d_m,
            "link_type": 1 if link_type.upper() == "LOS" else 0,
            "pl_dB": pl_dB,
            "p_tx_dBm": p_tx_dBm,
            "energy_J": energy_J,
        }

if __name__ == "__main__":
    # Prosty test manualny kompatybilności z BodyMap.
    bm = BodyMap()
    ch_model = ChannelModel(body_map=bm)

    # Używamy istniejącej metody sample_position zamiast random_sensor_position
    hub_pos = bm.default_hub_position()
    hub_region = "torso"
    sensor_pos = bm.sample_position("left_arm_lower")

    metrics = ch_model.link_metrics(
        p_tx=sensor_pos,
        region_tx="left_arm_lower",
        p_rx=hub_pos,
        region_rx=hub_region,
        packet_length_bits=8000,  # 1000 bajtów
        bit_rate_bps=250_000,     # 250 kbit/s
    )

    print("Metryki przykładowego łącza sensor–hub:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

