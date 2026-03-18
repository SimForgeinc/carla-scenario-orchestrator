from __future__ import annotations

import unittest
from unittest.mock import patch

from orchestrator.carla_metadata import CarlaMetadataService


class FakeBlueprint:
    def __init__(self, blueprint_id: str) -> None:
        self.id = blueprint_id


class FakeBlueprintLibrary:
    def filter(self, pattern: str):
        if pattern == 'vehicle.*':
            return [FakeBlueprint('vehicle.a'), FakeBlueprint('vehicle.b')]
        if pattern == 'walker.pedestrian.*':
            return [FakeBlueprint('walker.a')]
        return []


SIMPLE_XODR = """<OpenDRIVE>
  <road name="Road 1" length="20" id="1" junction="-1">
    <planView>
      <geometry s="0" x="0" y="0" hdg="0" length="20"><line /></geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <left>
          <lane id="1" type="driving"><width a="3.5" /></lane>
          <lane id="2" type="parking"><width a="2.0" /></lane>
        </left>
        <center><lane id="0" type="none" /></center>
        <right>
          <lane id="-1" type="driving"><width a="3.5" /></lane>
        </right>
      </laneSection>
    </lanes>
    <objects>
      <object id="10" name="Crosswalk" s="5" t="0" hdg="0">
        <outline>
          <cornerLocal u="0" v="0" />
          <cornerLocal u="2" v="0" />
          <cornerLocal u="2" v="3" />
          <cornerLocal u="0" v="3" />
        </outline>
      </object>
      <object id="11" name="StopSign" s="8" t="1" hdg="0" />
    </objects>
  </road>
</OpenDRIVE>"""


class FakeMap:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate_waypoints(self, distance: float):
        return []

    def to_opendrive(self):
        return SIMPLE_XODR


class FakeWorld:
    def __init__(self, current_map: str) -> None:
        self._map = FakeMap(current_map)

    def get_map(self):
        return self._map

    def get_blueprint_library(self):
        return FakeBlueprintLibrary()


class FakeClient:
    def __init__(self, current_map: str = 'Town01') -> None:
        self.current_map = current_map

    def get_world(self):
        return FakeWorld(self.current_map)

    def get_available_maps(self):
        return ['Town01', 'Town02_Opt']

    def get_server_version(self):
        return '0.9.16'

    def get_client_version(self):
        return '0.9.16'

    def load_world(self, map_name: str):
        self.current_map = map_name


class CarlaMetadataServiceTests(unittest.TestCase):
    def test_status_is_cached_until_invalidated(self) -> None:
        calls = {'count': 0}
        client = FakeClient()

        def fake_make_client(host, port, timeout):
            calls['count'] += 1
            return client

        with patch('orchestrator.carla_metadata._make_client', side_effect=fake_make_client):
            service = CarlaMetadataService(host='127.0.0.1', port=18689, timeout=20)
            first = service.get_status()
            second = service.get_status()
            self.assertEqual(calls['count'], 1)
            self.assertEqual(first.current_map, 'Town01')
            self.assertEqual(second.current_map, 'Town01')

            loaded = service.load_map('Town02_Opt')
            self.assertEqual(loaded.current_map, 'Town02_Opt')
            self.assertGreaterEqual(calls['count'], 2)

    def test_blueprints_are_cached(self) -> None:
        calls = {'count': 0}
        client = FakeClient()

        def fake_make_client(host, port, timeout):
            calls['count'] += 1
            return client

        with patch('orchestrator.carla_metadata._make_client', side_effect=fake_make_client):
            service = CarlaMetadataService(host='127.0.0.1', port=18689, timeout=20)
            first = service.list_blueprints()
            second = service.list_blueprints()
            self.assertEqual(calls['count'], 1)
            self.assertEqual(first['vehicles'], ['vehicle.a', 'vehicle.b'])
            self.assertEqual(second['walkers'], ['walker.a'])

    def test_stale_status_is_returned_while_refresh_is_deferred(self) -> None:
        calls = {'count': 0}
        client = FakeClient()

        def fake_make_client(host, port, timeout):
            calls['count'] += 1
            return client

        with patch('orchestrator.carla_metadata._make_client', side_effect=fake_make_client):
            service = CarlaMetadataService(host='127.0.0.1', port=18689, timeout=20)
            first = service.get_status(force_refresh=True)
            service._status_cache_ttl = -1.0
            with patch.object(service, '_refresh_status_cache_async') as refresh_cache:
                second = service.get_status()
            refresh_cache.assert_called_once()
            self.assertEqual(calls['count'], 1)
            self.assertEqual(first.current_map, 'Town01')
            self.assertEqual(second.current_map, 'Town01')

    def test_xodr_is_cached(self) -> None:
        calls = {'count': 0}
        client = FakeClient()

        def fake_make_client(host, port, timeout):
            calls['count'] += 1
            return client

        with patch('orchestrator.carla_metadata._make_client', side_effect=fake_make_client):
            service = CarlaMetadataService(host='127.0.0.1', port=18689, timeout=20)
            first = service.get_map_xodr()
            second = service.get_map_xodr()
            self.assertEqual(calls['count'], 1)
            self.assertIn("<OpenDRIVE>", first)
            self.assertEqual(first, second)

    def test_generated_map_is_cached(self) -> None:
        calls = {'count': 0}
        client = FakeClient()

        def fake_make_client(host, port, timeout):
            calls['count'] += 1
            return client

        with patch('orchestrator.carla_metadata._make_client', side_effect=fake_make_client):
            service = CarlaMetadataService(host='127.0.0.1', port=18689, timeout=20)
            first = service.get_generated_map()
            second = service.get_generated_map()
            self.assertEqual(calls['count'], 1)
            self.assertEqual(first['name'], 'Town01')
            self.assertEqual(first['stats']['roads'], 1)
            self.assertEqual(first['stats']['featureCounts']['parking'], 1)
            self.assertEqual(first['stats']['featureCounts']['crosswalk'], 1)
            self.assertEqual(len(first['crosswalks']), 1)
            self.assertEqual(len(first['stopMarkers']), 1)
            self.assertEqual(second['name'], 'Town01')


if __name__ == '__main__':
    unittest.main()
