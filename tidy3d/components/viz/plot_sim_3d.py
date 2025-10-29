from __future__ import annotations

from html import escape

from tidy3d.exceptions import SetupError


def plot_scene_3d(scene, width=800, height=800) -> None:
    import gzip
    import json
    from base64 import b64encode
    from io import BytesIO

    import h5py

    # Serialize scene to HDF5 in-memory
    buffer = BytesIO()
    scene.to_hdf5(buffer)
    buffer.seek(0)

    # Open source HDF5 for reading and prepare modified copy
    with h5py.File(buffer, "r") as src:
        buffer2 = BytesIO()
        with h5py.File(buffer2, "w") as dst:

            def copy_item(name, obj) -> None:
                if isinstance(obj, h5py.Group):
                    dst.create_group(name)
                    for k, v in obj.attrs.items():
                        dst[name].attrs[k] = v
                elif isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    if name == "JSON_STRING":
                        # Parse and update JSON string
                        json_str = (
                            data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
                        )
                        json_data = json.loads(json_str)
                        json_data["size"] = list(scene.size)
                        json_data["center"] = list(scene.center)
                        json_data["grid_spec"] = {}
                        new_str = json.dumps(json_data)
                        dst.create_dataset(name, data=new_str.encode("utf-8"))
                    else:
                        dst.create_dataset(name, data=data)
                        for k, v in obj.attrs.items():
                            dst[name].attrs[k] = v

            src.visititems(copy_item)
        buffer2.seek(0)

    # Gzip the modified HDF5
    gz_buffer = BytesIO()
    with gzip.GzipFile(fileobj=gz_buffer, mode="wb") as gz:
        gz.write(buffer2.read())
    gz_buffer.seek(0)

    # Base64 encode and display with gzipped flag
    sim_base64 = b64encode(gz_buffer.read()).decode("utf-8")
    plot_sim_3d(sim_base64, width=width, height=height, is_gz_base64=True)


def plot_sim_3d(sim, width=800, height=800, is_gz_base64=False) -> None:
    """Make 3D display of simulation in ipython notebook."""

    try:
        from IPython.display import HTML, display
    except ImportError as e:
        raise SetupError(
            "3D plotting requires ipython to be installed "
            "and the code to be running on a jupyter notebook."
        ) from e

    from base64 import b64encode
    from io import BytesIO

    if not is_gz_base64:
        buffer = BytesIO()
        sim.to_hdf5_gz(buffer)
        buffer.seek(0)
        base64 = b64encode(buffer.read()).decode("utf-8")
    else:
        base64 = sim

    js_code = """
        /**
        * Simulation Viewer Injector
        *
        * Monitors the document for elements being added in the form:
        *
        *    <div class="simulation-viewer" data-width="800" data-height="800" data-simulation="{...}" />
        *
        * This script will then inject an iframe to the viewer application, and pass it the simulation data
        * via the postMessage API on request. The script may be safely included multiple times, with only the
        * configuration of the first started script (e.g. viewer URL) applying.
        *
        */
        (function() {
            const TARGET_CLASS = "simulation-viewer";
            const ACTIVE_CLASS = "simulation-viewer-active";
            const VIEWER_URL = "https://tidy3d.simulation.cloud/simulation-viewer";

            class SimulationViewerInjector {
                constructor() {
                    for (var node of document.getElementsByClassName(TARGET_CLASS)) {
                        this.injectViewer(node);
                    }

                    // Monitor for newly added nodes to the DOM
                    this.observer = new MutationObserver(this.onMutations.bind(this));
                    this.observer.observe(document.body, {childList: true, subtree: true});
                }

                onMutations(mutations) {
                    for (var mutation of mutations) {
                        if (mutation.type === 'childList') {
                            /**
                            * Have found that adding the element does not reliably trigger the mutation observer.
                            * It may be the case that setting content with innerHTML does not trigger.
                            *
                            * It seems to be sufficient to re-scan the document for un-activated viewers
                            * whenever an event occurs, as Jupyter triggers multiple events on cell evaluation.
                            */
                            var viewers = document.getElementsByClassName(TARGET_CLASS);
                            for (var node of viewers) {
                                this.injectViewer(node);
                            }
                        }
                    }
                }

                injectViewer(node) {
                    // (re-)check that this is a valid simulation container and has not already been injected
                    if (node.classList.contains(TARGET_CLASS) && !node.classList.contains(ACTIVE_CLASS)) {
                        // Mark node as injected, to prevent re-runs
                        node.classList.add(ACTIVE_CLASS);

                        var uuid;
                        if (window.crypto && window.crypto.randomUUID) {
                            uuid = window.crypto.randomUUID();
                        } else {
                            uuid = "" + Math.random();
                        }

                        var frame = document.createElement("iframe");
                        frame.width = node.dataset.width || 800;
                        frame.height = node.dataset.height || 800;
                        frame.style.cssText = `width:${frame.width}px;height:${frame.height}px;max-width:none;border:0;display:block`
                        frame.src = VIEWER_URL + "?uuid=" + uuid;

                        var postMessageToViewer;
                        postMessageToViewer = event => {
                            if(event.data.type === 'viewer' && event.data.uuid===uuid){
                                frame.contentWindow.postMessage({ type: 'jupyter', uuid, value: node.dataset.simulation, fileType: 'hdf5'}, '*');

                                // Run once only
                                window.removeEventListener('message', postMessageToViewer);
                            }
                        };
                        window.addEventListener(
                            'message',
                            postMessageToViewer,
                            false
                        );

                        node.appendChild(frame);
                    }
                }
            }

            if (!window.simulationViewerInjector) {
                window.simulationViewerInjector = new SimulationViewerInjector();
            }
        })();
    """
    html_code = f"""
    <div class="simulation-viewer" data-width="{escape(str(width))}" data-height="{escape(str(height))}" data-simulation="{escape(base64)}" ></div>
    <script>
        {js_code}
    </script>
    """

    return display(HTML(html_code))
