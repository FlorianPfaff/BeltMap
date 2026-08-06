from beltmap import _driver_runtime as rt


def test_ru_maxrss_to_mib_uses_kib_units_on_linux():
    assert rt._ru_maxrss_to_mib(2048, platform="linux") == 2.0


def test_ru_maxrss_to_mib_uses_byte_units_on_macos():
    assert rt._ru_maxrss_to_mib(2 * 1024 * 1024, platform="darwin") == 2.0
