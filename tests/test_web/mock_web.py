# custom class to be the mock return value
# will override the requests.Response returned from requests.get
class MockResponse:
    def __init__(self, code, json_data):
        self.status_code = code
        self.json_data = json_data

    def json(self):
        return self.json_data
