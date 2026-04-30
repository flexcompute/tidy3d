(function () {
  // The paired flex docs workflow publishes this project-root manifest.
  const PRODUCTION_DOCS_ROOT_PATH = "/projects/tidy3d/en/";
  const PRODUCTION_MANIFEST_URL = "/projects/tidy3d/versions.json";
  const SCRIPT_ELEMENT =
    document.currentScript ||
    document.querySelector(
      'script[src$="/version-switcher.js"], script[src$="version-switcher.js"]'
    );

  function normalizePath(path) {
    return path.endsWith("/") ? path : `${path}/`;
  }

  function currentScriptUrl() {
    if (!SCRIPT_ELEMENT || !SCRIPT_ELEMENT.src) {
      return null;
    }
    return new URL(SCRIPT_ELEMENT.src, window.location.href);
  }

  function docsRootUrl() {
    if (window.TIDY3D_DOCS_ROOT_URL) {
      return new URL(window.TIDY3D_DOCS_ROOT_URL, window.location.href);
    }
    if (window.location.pathname.startsWith(PRODUCTION_DOCS_ROOT_PATH)) {
      return new URL(PRODUCTION_DOCS_ROOT_PATH, window.location.origin);
    }

    const scriptUrl = currentScriptUrl();
    if (scriptUrl) {
      return new URL("../../", scriptUrl);
    }

    return new URL("./", window.location.href);
  }

  function manifestUrl(docsRoot) {
    if (window.TIDY3D_DOCS_VERSIONS_URL) {
      return new URL(window.TIDY3D_DOCS_VERSIONS_URL, window.location.href).toString();
    }
    if (window.location.pathname.startsWith(PRODUCTION_DOCS_ROOT_PATH)) {
      return PRODUCTION_MANIFEST_URL;
    }
    return new URL("versions.json", docsRoot).toString();
  }

  function buildTargetPath(version) {
    return `${version}/`;
  }

  function entryUrl(entry, docsRoot) {
    return new URL(entry.path || buildTargetPath(entry.name), docsRoot);
  }

  function entryPath(entry, docsRoot) {
    return entryUrl(entry, docsRoot).pathname;
  }

  function parseCurrentVersion(pathname, versions, docsRoot) {
    const currentPath = normalizePath(pathname);
    const matchingEntry = versions
      .filter((entry) => entry && entry.name)
      .sort(
        (left, right) => entryPath(right, docsRoot).length - entryPath(left, docsRoot).length
      )
      .find((entry) =>
        currentPath.startsWith(normalizePath(entryPath(entry, docsRoot)))
      );

    return matchingEntry ? matchingEntry.name : null;
  }

  function findMountPoint() {
    return (
      document.querySelector(".bd-sidebar-primary .sidebar-primary-items__end") ||
      document.querySelector(".bd-sidebar-primary .sidebar-primary-items") ||
      document.querySelector(".bd-sidebar-primary") ||
      document.querySelector(".bd-sidebar")
    );
  }

  function renderSwitcher(manifest, currentVersion, docsRoot) {
    const mountPoint = findMountPoint();
    if (!mountPoint || mountPoint.querySelector(".tidy3d-version-switcher")) {
      return;
    }

    const versions = Array.isArray(manifest.versions) ? manifest.versions : [];
    if (!versions.length) {
      return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "tidy3d-version-switcher";

    const label = document.createElement("label");
    label.className = "tidy3d-version-switcher__label";
    label.htmlFor = "tidy3d-version-switcher-select";
    label.textContent = "Version";

    const select = document.createElement("select");
    select.id = "tidy3d-version-switcher-select";
    select.className = "tidy3d-version-switcher__select";

    versions.forEach((entry) => {
      if (!entry || !entry.name) {
        return;
      }

      const option = document.createElement("option");
      option.value = entryUrl(entry, docsRoot).toString();
      option.textContent = entry.label || entry.name;
      option.selected = entry.name === currentVersion;
      select.appendChild(option);
    });

    select.addEventListener("change", function () {
      window.location.assign(select.value);
    });

    wrapper.appendChild(label);
    wrapper.appendChild(select);
    mountPoint.appendChild(wrapper);
  }

  async function initVersionSwitcher() {
    const docsRoot = docsRootUrl();
    try {
      const response = await fetch(manifestUrl(docsRoot), { cache: "no-store" });
      if (!response.ok) {
        return;
      }

      const manifest = await response.json();
      const versions = Array.isArray(manifest.versions) ? manifest.versions : [];
      const currentVersion = parseCurrentVersion(
        window.location.pathname,
        versions,
        docsRoot
      );
      renderSwitcher(manifest, currentVersion, docsRoot);
    } catch (error) {
      console.warn("Unable to load the Tidy3D docs version manifest.", error);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initVersionSwitcher);
  } else {
    initVersionSwitcher();
  }
})();
