from __future__ import annotations

from html import escape

from tidy3d.exceptions import SetupError


def plot_sim_3d(sim, width=800, height=800) -> None:
    """Make 3D display of simulation in ipyython notebook."""

    try:
        from IPython.display import HTML, display
    except ImportError as e:
        raise SetupError(
            "3D plotting requires ipython to be installed "
            "and the code to be running on a jupyter notebook."
        ) from e

    from base64 import b64encode
    from io import BytesIO

    buffer = BytesIO()
    sim.to_hdf5_gz(buffer)
    buffer.seek(0)
    base64 = b64encode(buffer.read()).decode("utf-8")
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
