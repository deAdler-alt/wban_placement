# ../model/energy_model.py

"""
Model energii i funkcji celu F(g) dla rozmieszczenia sensorów WBAN.

Ten moduł:
- dekoduje wektor decyzji g -> pozycje sensorów + pozycja CH,
- sprawdza ograniczenia geometryczne (Ω_i, Ω_CH),
- korzysta z ChannelModel do policzenia:
    * strat ścieżki PL_ij(g),
    * wymaganej mocy nadawania P_TX_req^{ij}(g),
    * energii transmisji E_TX^{ij}(g),
- wylicza:
    * całkowitą energię na rundę E_total(g),
    * minimalny margines łącza M_min(g),
    * składowe f_E(g), f_M(g) i globalną funkcję celu F(g),
zgodnie z rozdziałem 3.2–3.3 pracy.

Uwaga: funkcja celu jest zdefiniowana jako MINIMIZACYJNA (im mniejsza, tym lepsze
rozmieszczenie). W DEAP używamy FitnessMin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Dict, Optional, Tuple

from body_map import BodyMap, Point
from channel_model import ChannelModel


@dataclass
class EnergyModelConfig:
    """
    Konfiguracja modelu energii i funkcji celu.

    Atrybuty:
        num_sensors:
            Liczba sensorów mobilnych k (zgodnie z rozdziałem 3.3).
        sensor_regions:
            Lista nazw regionów anatomicznych Ω_i dla każdego sensora
            (np. ["torso", "left_arm_lower", ...]), długości num_sensors.
        ch_region:
            Region, w którym musi leżeć koncentrator (CH), domyślnie "torso".
        bs_region:
            Region odpowiadający pozycji bramki / huba (BS) w BodyMap,
            wykorzystywany do klasyfikacji łącza CH–BS (np. "torso").
        packet_length_bits:
            Długość pakietu generowanego przez pojedynczy sensor (SN) [bity].
        aggregation_factor:
            Współczynnik agregacji w CH. Pakiet CH->BS ma długość:
                L_CHBS = aggregation_factor * num_sensors * packet_length_bits.
            Dla braku kompresji: aggregation_factor = 1.0.
        bit_rate_bps:
            Przepływność łącza [bity/s].
        tx_power_max_dBm:
            Maksymalna dostępna moc nadawania P_TX,max [dBm].
            Używana zarówno jako ograniczenie (brak zasięgu -> kara),
            jak i do liczenia marginesu łącza.
        energy_ref_J:
            Energia referencyjna E_ref [J] do normalizacji f_E(g).
            Jeżeli nie jest znana, można przyjąć 1.0 (skaluje funkcję celu).
        w_energy:
            Waga składowej energetycznej f_E(g) w funkcji F(g).
        w_margin:
            Waga składowej marginesu łącza f_M(g) w funkcji F(g).
            Dla spójności zwykle w_energy + w_margin = 1.
        margin_target_dB:
            Docelowy minimalny margines łącza M_target [dB].
            Jeżeli M_min(g) >= M_target, f_M(g) = 0.
        penalty_invalid:
            Kara (duża wartość funkcji celu) dla rozwiązań naruszających
            ograniczenia łączności (P_TX_req > P_TX_max).
        penalty_geom:
            Kara dla rozwiązań naruszających ograniczenia geometryczne
            (poza regionem Ω_i lub Ω_CH).
    """
    num_sensors: int
    sensor_regions: List[str]
    ch_region: str = "torso"
    bs_region: str = "torso"
    packet_length_bits: int = 8000
    aggregation_factor: float = 1.0
    bit_rate_bps: float = 250_000.0
    tx_power_max_dBm: float = 0.0
    energy_ref_J: float = 1.0
    w_energy: float = 0.7
    w_margin: float = 0.3
    margin_target_dB: float = 10.0
    penalty_invalid: float = 1e6
    penalty_geom: float = 1e6

    def __post_init__(self) -> None:
        if len(self.sensor_regions) != self.num_sensors:
            raise ValueError(
                f"len(sensor_regions)={len(self.sensor_regions)} "
                f"!= num_sensors={self.num_sensors}"
            )
        if self.energy_ref_J <= 0.0:
            raise ValueError("energy_ref_J must be positive.")
        if not (0.0 <= self.w_energy <= 1.0 and 0.0 <= self.w_margin <= 1.0):
            raise ValueError("w_energy and w_margin must be in [0, 1].")


class EnergyModel:
    """
    Klasa implementująca model energii i funkcji celu F(g).

    Użycie typowe:
        bm = BodyMap()
        ch_model = ChannelModel(body_map=bm)
        cfg = EnergyModelConfig(...)
        em = EnergyModel(config=cfg, body_map=bm, channel_model=ch_model)

        fitness_tuple = em.evaluate(genome)   # zgodnie z DEAP (zwraca (F,))
    """

    def __init__(
        self,
        config: EnergyModelConfig,
        body_map: Optional[BodyMap] = None,
        channel_model: Optional[ChannelModel] = None,
    ) -> None:
        self.config = config
        self.body_map = body_map or BodyMap()
        self.channel_model = channel_model or ChannelModel(body_map=self.body_map)

    # ------------------------------------------------------------------
    # Dekodowanie wektora decyzji g
    # ------------------------------------------------------------------
    def decode_genome(self, genome: Sequence[float]) -> Tuple[List[Point], Point]:
        """
        Dekoduje wektor g na:
            - listę pozycji sensorów [p_1, ..., p_k],
            - pozycję koncentratora p_CH.

        Oczekiwana długość wektora:
            len(g) = 2 * k + 2
        gdzie k = num_sensors.
        """
        expected_len = 2 * self.config.num_sensors + 2
        if len(genome) != expected_len:
            raise ValueError(
                f"Genome length {len(genome)} does not match expected {expected_len} "
                f"for num_sensors = {self.config.num_sensors}."
            )

        coords = list(genome)
        sensor_positions: List[Point] = []
        for i in range(self.config.num_sensors):
            x = float(coords[2 * i])
            y = float(coords[2 * i + 1])
            sensor_positions.append((x, y))

        ch_x = float(coords[2 * self.config.num_sensors])
        ch_y = float(coords[2 * self.config.num_sensors + 1])
        ch_pos: Point = (ch_x, ch_y)

        return sensor_positions, ch_pos

    # ------------------------------------------------------------------
    # Sprawdzanie ograniczeń geometrycznych
    # ------------------------------------------------------------------
    def _check_geometry(
        self,
        sensor_positions: List[Point],
        ch_pos: Point,
    ) -> bool:
        """
        Sprawdza, czy:
          - każdy sensor i leży w swoim regionie Ω_i,
          - CH leży w regionie Ω_CH.

        Zwraca True, jeżeli wszystkie ograniczenia są spełnione.
        """
        # Sensory
        for i, p in enumerate(sensor_positions):
            region = self.config.sensor_regions[i]
            if not self.body_map.is_inside_region(p, region):
                return False

        # Koncentrator (CH)
        if not self.body_map.is_inside_region(ch_pos, self.config.ch_region):
            return False

        return True

    # ------------------------------------------------------------------
    # Główne obliczenia: energia, margines łącza, funkcja celu
    # ------------------------------------------------------------------
    def compute_metrics(self, genome: Sequence[float]) -> Dict[str, float]:
        """
        Oblicza komplet metryk dla danego rozmieszczenia g.

        Zwraca słownik zawierający m.in.:
            - fitness      : wartość funkcji celu F(g) (do minimizacji),
            - valid        : 1.0 jeśli spełniono wszystkie ograniczenia, 0.0 w p.p.,
            - E_total      : całkowita energia transmisji w jednej rundzie [J],
            - f_E          : składowa energetyczna funkcji celu,
            - margin_min   : minimalny margines łącza [dB],
            - f_margin     : składowa związana z marginesem,
            - p_tx_req_max : maksymalna wymagana moc nadawania [dBm].
        """
        sensor_positions, ch_pos = self.decode_genome(genome)

        # 1. Ograniczenia geometryczne (Ω_i, Ω_CH)
        geometry_ok = self._check_geometry(sensor_positions, ch_pos)
        if not geometry_ok:
            # Niepoprawna geometria → kara
            return {
                "fitness": self.config.penalty_geom,
                "valid": 0.0,
                "E_total": float("nan"),
                "f_E": float("nan"),
                "margin_min": float("-inf"),
                "f_margin": 1.0,
                "p_tx_req_max": float("nan"),
            }

        k = self.config.num_sensors
        L_sn = self.config.packet_length_bits
        # Długość pakietu CH->BS (zakładamy brak kompresji: factor * k)
        L_ch_bs = int(self.config.aggregation_factor * k * L_sn)

        p_tx_req_values: List[float] = []
        energies_J: List[float] = []

        # 2. Łącza SN_i -> CH
        for i, p_sn in enumerate(sensor_positions):
            region_sn = self.config.sensor_regions[i]
            metrics_link = self.channel_model.link_metrics(
                p_tx=p_sn,
                region_tx=region_sn,
                p_rx=ch_pos,
                region_rx=self.config.ch_region,
                packet_length_bits=L_sn,
                bit_rate_bps=self.config.bit_rate_bps,
            )
            p_tx_req_values.append(metrics_link["p_tx_dBm"])
            energies_J.append(metrics_link["energy_J"])

        # 3. Łącze CH -> BS (hub)
        metrics_ch_bs = self.channel_model.link_metrics(
            p_tx=ch_pos,
            region_tx=self.config.ch_region,
            p_rx=self.body_map.hub_pos,
            region_rx=self.config.bs_region,
            packet_length_bits=L_ch_bs,
            bit_rate_bps=self.config.bit_rate_bps,
        )
        p_tx_req_values.append(metrics_ch_bs["p_tx_dBm"])
        energies_J.append(metrics_ch_bs["energy_J"])

        # 4. Ograniczenie mocy nadawania (P_TX_req <= P_TX_max) dla wszystkich łączy
        p_tx_req_max = max(p_tx_req_values)
        if p_tx_req_max > self.config.tx_power_max_dBm:
            # brak zasięgu na choć jednym łączu → rozwiązanie niepoprawne
            margin_min = self.config.tx_power_max_dBm - p_tx_req_max
            return {
                "fitness": self.config.penalty_invalid,
                "valid": 0.0,
                "E_total": sum(energies_J),
                "f_E": float("nan"),
                "margin_min": margin_min,
                "f_margin": 1.0,
                "p_tx_req_max": p_tx_req_max,
            }

        # 5. Całkowita energia transmisji w jednej rundzie
        E_total = sum(energies_J)
        f_E = E_total / self.config.energy_ref_J

        # 6. Margines łącza i składowa jakości (f_M)
        margins_dB = [
            self.config.tx_power_max_dBm - p_req for p_req in p_tx_req_values
        ]
        margin_min = min(margins_dB)

        if margin_min >= self.config.margin_target_dB:
            f_margin = 0.0
        else:
            deficit = self.config.margin_target_dB - margin_min
            f_margin = deficit / self.config.margin_target_dB

        # 7. Ostateczna funkcja celu (minimizacyjna)
        F = self.config.w_energy * f_E + self.config.w_margin * f_margin

        return {
            "fitness": F,
            "valid": 1.0,
            "E_total": E_total,
            "f_E": f_E,
            "margin_min": margin_min,
            "f_margin": f_margin,
            "p_tx_req_max": p_tx_req_max,
        }

    # ------------------------------------------------------------------
    # Interfejs przyjazny dla DEAP: evaluate() zwraca krotkę (F,)
    # ------------------------------------------------------------------
    def evaluate(self, genome: Sequence[float]) -> Tuple[float]:
        """
        Funkcja zgodna z konwencją DEAP:
            toolbox.evaluate(individual) -> (fitness,)

        Zwraca:
            (F(g),) – wartość funkcji celu do MINIMIZACJI.
        """
        metrics = self.compute_metrics(genome)
        return (metrics["fitness"],)


# ----------------------------------------------------------------------
# Prosty test manualny modułu
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import random

    # Konfiguracja przykładowego scenariusza
    body = BodyMap()
    ch_model = ChannelModel(body_map=body)

    sensor_regions = ["torso", "left_arm_lower", "right_arm_lower", "left_leg"]
    cfg = EnergyModelConfig(
        num_sensors=len(sensor_regions),
        sensor_regions=sensor_regions,
        ch_region="torso",
        bs_region="torso",
        packet_length_bits=8000,
        aggregation_factor=1.0,
        bit_rate_bps=250_000.0,
        tx_power_max_dBm=0.0,   # 1 mW – typowa moc WBAN
        energy_ref_J=1.0,
        w_energy=0.7,
        w_margin=0.3,
        margin_target_dB=10.0,
        penalty_invalid=1e6,
        penalty_geom=1e6,
    )

    energy_model = EnergyModel(config=cfg, body_map=body, channel_model=ch_model)

    # Losowa, poprawna geometria:
    rng = random.Random(123)
    genome: List[float] = []

    for region in sensor_regions:
        genome.extend(body.sample_position(region_name=region, rng=rng))

    # CH w okolicy klatki (torso)
    genome.extend(body.sample_position(region_name="torso", rng=rng))

    metrics = energy_model.compute_metrics(genome)
    print("Przykładowe metryki dla losowego rozmieszczenia:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nWartość fitness (DEAP):", energy_model.evaluate(genome))
