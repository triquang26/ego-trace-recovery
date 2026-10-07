# Repo rules

- Code không comment, không docstring, không file NOTE/HISTORY/CHANGELOG.
- Không monkeypatch; thay thành phần bằng dependency injection qua constructor/tham số.
- Mỗi file Python tối đa 200 dòng; tách module khi vượt.
- Không commit token, key hay credentials.
- Chạy `pytest` trước khi commit.
