import pytest
import genesis as gs
import numpy as np

@pytest.fixture
def setup_scene():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=10,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=False,
    )
    return scene

def test_runtime_environment_switching(setup_scene):
    scene = setup_scene

    # Initialize with CPU backend
    gs.init(backend=gs.cpu)
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()
    for _ in range(100):
        scene.step()

    # Switch to GPU backend
    gs.destroy()
    gs.init(backend=gs.gpu)
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()
    for _ in range(100):
        scene.step()

    assert True  # If no exceptions, the test passes

def test_circumvention_mechanism(setup_scene):
    scene = setup_scene

    # Initialize with CPU backend
    gs.init(backend=gs.cpu)
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()
    for _ in range(100):
        scene.step()

    # Circumvent by changing simulation options
    scene.sim_options.dt = 0.02
    for _ in range(100):
        scene.step()

    assert True  # If no exceptions, the test passes

def test_edge_cases(setup_scene):
    scene = setup_scene

    # Initialize with CPU backend
    gs.init(backend=gs.cpu)
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()

    # Test extreme values
    plane.set_dofs_position(np.array([1e10, 1e10, 1e10]))
    for _ in range(10):
        scene.step()

    # Test boundary conditions
    plane.set_dofs_position(np.array([0, 0, 0]))
    for _ in range(10):
        scene.step()

    # Test invalid inputs
    with pytest.raises(ValueError):
        plane.set_dofs_position(np.array([None, None, None]))

    assert True  # If no exceptions, the test passes
