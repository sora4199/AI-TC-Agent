from datetime import datetime


class DateTime:

    def get_datetime(self):
        now = datetime.now().astimezone()

        return {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
        }


if __name__ == "__main__":
    tool = DateTime()

    result = tool.get_datetime()

    print(result)