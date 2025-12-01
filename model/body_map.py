# ../model/body_map.py

"""
Moduł odpowiada za geometryczną reprezentację ciała w 2D oraz podstawowe
operacje na pozycjach sensorów:
- definicja prostokątnych regionów anatomicznych (Ω_i),
- losowanie pozycji w danym regionie,
- sprawdzanie, czy punkt należy do regionu,
- klasyfikacja łącza jako LOS / NLOS na podstawie regionów.

Ten moduł NIE zna funkcji celu ani DEAP – jest czystą geometrią, którą
wykorzysta później channel_model.py i fitness.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable
import math
import random


Point = Tuple[float, float]


@dataclass(frozen=True)
class RectRegion:
    """
    Prosty, osiowo wyrównany prostokątny region na mapie ciała.

    Wszystkie współrzędne są w metrach w lokalnym układzie odniesienia:
    x ∈ [0, body_width], y ∈ [0, body_height].

    Attributes:
        name: Nazwa regionu, np. "torso", "left_arm".
        xmin, xmax, ymin, ymax: Granice prostokąta.
    """
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, p: Point) -> bool:
        """Sprawdza, czy punkt p = (x, y) leży wewnątrz regionu (łącznie z brzegiem)."""
        x, y = p
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)

    def clamp(self, p: Point) -> Point:
        """
        Projekcja punktu p na prostokąt – jeśli punkt wychodzi poza region,
        zostaje "przycięty" do najbliższej krawędzi.
        """
        x, y = p
        x_clamped = min(max(x, self.xmin), self.xmax)
        y_clamped = min(max(y, self.ymin), self.ymax)
        return x_clamped, y_clamped

    def sample_uniform(self, rng: Optional[random.Random] = None) -> Point:
        """Losuje punkt z rozkładu jednostajnego wewnątrz prostokąta."""
        if rng is None:
            rng = random
        x = rng.uniform(self.xmin, self.xmax)
        y = rng.uniform(self.ymin, self.ymax)
        return x, y


class BodyMap:
    """
    Abstrakcyjna mapa ciała w 2D z prostokątnymi regionami anatomicznymi.

    Założenia:
    - Używamy znormalizowanego układu współrzędnych:
      szerokość ciała ≈ 0.4 m, wysokość ≈ 1.8 m (umowne wartości).
    - (0, 0) leży przy spodzie sylwetki, (body_width, body_height) przy czubku głowy.
    - Hub (BS) jest domyślnie umieszczony w okolicy pasa z przodu tułowia.

    Ten model jest uproszczony, ale wystarczający do:
    - generowania sensownych pozycji startowych dla sensorów,
    - klasyfikacji łączy jako LOS/NLOS na potrzeby modelu kanału.
    """

    def __init__(
        self,
        body_width: float = 0.4,
        body_height: float = 1.8,
        hub_pos: Optional[Point] = None,
        regions: Optional[Dict[str, RectRegion]] = None,
    ) -> None:
        self.body_width = body_width
        self.body_height = body_height

        # Domyślna pozycja huba – okolice pasa na środku ciała
        if hub_pos is None:
            hub_pos = (body_width / 2.0, body_height * 0.9)
        self.hub_pos: Point = hub_pos

        # Jeśli nie podano regionów – tworzymy sensowny domyślny podział
        if regions is None:
            regions = self._create_default_regions()
        self.regions: Dict[str, RectRegion] = regions

    # ------------------------------------------------------------------
    # Definicja domyślnych regionów anatomicznych
    # ------------------------------------------------------------------

    def _create_default_regions(self) -> Dict[str, RectRegion]:
        """
        Tworzy prosty, domyślny zestaw regionów (tułów + kończyny)
        na podstawie wymiarów ciała.
        """
        w = self.body_width
        h = self.body_height

        regions = {
            # Tułów (przód) – główna "płyta" na klatce piersiowej
            "torso": RectRegion(
                name="torso",
                xmin=0.1 * w,
                xmax=0.9 * w,
                ymin=0.5 * h,
                ymax=1.3 * h,
            ),
            # Głowa
            "head": RectRegion(
                name="head",
                xmin=0.25 * w,
                xmax=0.75 * w,
                ymin=1.3 * h,
                ymax=1.6 * h,
            ),
            # Lewa ręka (górna część)
            "left_arm_upper": RectRegion(
                name="left_arm_upper",
                xmin=-0.1 * w,
                xmax=0.1 * w,
                ymin=0.8 * h,
                ymax=1.2 * h,
            ),
            # Lewa ręka (dół / nadgarstek)
            "left_arm_lower": RectRegion(
                name="left_arm_lower",
                xmin=-0.15 * w,
                xmax=0.05 * w,
                ymin=0.4 * h,
                ymax=0.8 * h,
            ),
            # Prawa ręka (górna część)
            "right_arm_upper": RectRegion(
                name="right_arm_upper",
                xmin=0.9 * w,
                xmax=1.1 * w,
                ymin=0.8 * h,
                ymax=1.2 * h,
            ),
            # Prawa ręka (dół / nadgarstek)
            "right_arm_lower": RectRegion(
                name="right_arm_lower",
                xmin=0.95 * w,
                xmax=1.25 * w,
                ymin=0.4 * h,
                ymax=0.8 * h,
            ),
            # Lewa noga
            "left_leg": RectRegion(
                name="left_leg",
                xmin=0.15 * w,
                xmax=0.35 * w,
                ymin=0.0 * h,
                ymax=0.5 * h,
            ),
            # Prawa noga
            "right_leg": RectRegion(
                name="right_leg",
                xmin=0.65 * w,
                xmax=0.85 * w,
                ymin=0.0 * h,
                ymax=0.5 * h,
            ),
        }

        return regions

    # ------------------------------------------------------------------
    # Dostęp do regionów
    # ------------------------------------------------------------------

    def get_region(self, name: str) -> RectRegion:
        """Zwraca region o zadanej nazwie lub podnosi KeyError, jeśli nie istnieje."""
        return self.regions[name]

    def list_regions(self) -> Iterable[str]:
        """Zwraca listę dostępnych nazw regionów."""
        return self.regions.keys()

    # ------------------------------------------------------------------
    # Operacje na pozycjach sensorów
    # ------------------------------------------------------------------

    def is_inside_region(self, p: Point, region_name: str) -> bool:
        """Sprawdza, czy punkt p należy do wskazanego regionu."""
        region = self.get_region(region_name)
        return region.contains(p)

    def clamp_to_region(self, p: Point, region_name: str) -> Point:
        """
        Projekcja punktu p do wnętrza regionu.

        Użyteczne np. do naprawy punktów, które po mutacji w GA wypadły
        poza dozwoloną dziedzinę.
        """
        region = self.get_region(region_name)
        return region.clamp(p)

    def sample_position(self, region_name: str, rng: Optional[random.Random] = None) -> Point:
        """Losuje pozycję w zadanym regionie anatomicznym."""
        region = self.get_region(region_name)
        return region.sample_uniform(rng=rng)

    @staticmethod
    def euclidean_distance(p1: Point, p2: Point) -> float:
        """Zwykła odległość euklidesowa między dwoma punktami 2D."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # NEW: metoda instancyjna distance używana przez ChannelModel
    def distance(self, p1: Point, p2: Point) -> float:
        """Alias dla euclidean_distance, w formie metody instancyjnej."""
        return self.euclidean_distance(p1, p2)

    # NEW: domyślna pozycja huba używana w channel_model.__main__
    def default_hub_position(self) -> Point:
        """Zwraca domyślną pozycję huba/stacji bazowej."""
        return self.hub_pos

    # ------------------------------------------------------------------
    # Heurystyczna klasyfikacja łącza jako LOS / NLOS
    # ------------------------------------------------------------------

    def classify_link_los_nlos(self, p1: Point, region1: str, p2: Point, region2: str) -> str:
        """
        Heurystycznie klasyfikuje łącze między dwoma punktami jako LOS lub NLOS,
        na podstawie regionów, do których należą.

        Założenia (uproszczone, ale zgodne z intuicją z literatury):
        - Połączenia w obrębie tej samej kończyny lub tułowia są traktowane jako LOS.
        - Połączenia "przez tors", czyli np. między nogą a ręką z przeciwnej strony,
          są traktowane jako NLOS (silne tłumienie).
        """
        if region1 == region2:
            return "LOS"

        # Grupy regionów, które traktujemy jako "na tej samej części ciała"
        torso_group = {"torso", "head"}
        left_limb_group = {"left_arm_upper", "left_arm_lower", "left_leg"}
        right_limb_group = {"right_arm_upper", "right_arm_lower", "right_leg"}

        def group_of(region: str) -> str:
            if region in torso_group:
                return "torso"
            if region in left_limb_group:
                return "left"
            if region in right_limb_group:
                return "right"
            return "other"

        g1 = group_of(region1)
        g2 = group_of(region2)

        # Ta sama grupa → raczej LOS (np. wzdłuż tej samej kończyny)
        if g1 == g2 and g1 != "other":
            return "LOS"

        # Torso ↔ kończyna po tej samej stronie – LOS (np. klatka ↔ lewe ramię)
        if (g1 == "torso" and g2 in {"left", "right"}) or (g2 == "torso" and g1 in {"left", "right"}):
            # sprawdzamy, czy to ta sama strona
            if (g1 == "torso" and g2 == "left") or (g2 == "torso" and g1 == "left"):
                return "LOS"
            if (g1 == "torso" and g2 == "right") or (g2 == "torso" and g1 == "right"):
                return "LOS"

        # Wszystkie pozostałe przypadki traktujemy jako NLOS (konserwatywnie)
        return "NLOS"

    # NEW: prosty alias, którego oczekuje ChannelModel.classify_link(...)
    def classify_link(self, p1: Point, region1: str, p2: Point, region2: str) -> str:
        """
        Alias do classify_link_los_nlos, zachowujący interfejs oczekiwany
        przez ChannelModel.
        """
        return self.classify_link_los_nlos(p1, region1, p2, region2)


# ----------------------------------------------------------------------
# Przykładowe użycie modułu (do szybkich testów ręcznych)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(42)
    body = BodyMap()

    print("Dostępne regiony:", list(body.list_regions()))
    print("Domyślna pozycja huba:", body.default_hub_position())

    # Przykład losowania pozycji dla sensora na lewym nadgarstku
    p_sensor = body.sample_position("left_arm_lower", rng=rng)
    print("Przykładowa pozycja sensora (left_arm_lower):", p_sensor)

    # Sprawdzenie LOS/NLOS dla połączenia sensor ↔ hub
    link_type = body.classify_link(
        p1=p_sensor,
        region1="left_arm_lower",
        p2=body.default_hub_position(),
        region2="torso",
    )
    print("Typ łącza sensor–hub:", link_type)
