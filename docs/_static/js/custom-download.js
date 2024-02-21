(function () {

  function getFileAbsoluteUrl(url) {
    return new URL(url, window.location.href);;
  }

  function _xhrDownload(url, fileName) {
    let xhr = new XMLHttpRequest();
    xhr.open('get', url, true);
    xhr.setRequestHeader('Cache-Control', 'no-cache');
    xhr.setRequestHeader('Content-type', 'application/ipynb');
    xhr.responseType = 'blob';
    xhr.onload = function () {
      if (this.status === 200) {
        let blob = this.response;
        let url = window.URL.createObjectURL(blob);
        let a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        a.click();
      }
    };
    xhr.onerror = function () {
      console.error('error occurred when download file');
    };
    xhr.send();
  }

  function handleIpynbDownload(url) {
    const fileName = url.split("/").pop();
    _xhrDownload(url, fileName)
  }

  function enhanceDownload() {
    window.onload = function () {
      setTimeout(() => {
        const aTags = document.querySelectorAll(
          'a[data-original-title="Download source file"]'
        );
        // Change the downloading behavior of <a> elements.
        // Remove the href attribute and add an onclick event.
        aTags.forEach((aTag) => {
          aTag.setAttribute("download", "");
          const href = aTag.getAttribute("href")
          const absoluteUrl = getFileAbsoluteUrl(href);
          aTag.setAttribute("onclick", "handleIpynbDownload('" + absoluteUrl + "')")
          aTag.setAttribute("href", "javascript:void(0)")
        });
      }, 1000);
    };
  }

  enhanceDownload()

  window.handleIpynbDownload = handleIpynbDownload

})()
