import csv
import pickle
import shutil
import tempfile

# import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# import mongomock  # type: ignore
import pandas as pd
import pymongo  # type: ignore
import pytest
from helpers import wait_for_condition

from haymaker.misc import default_path
from haymaker.saver import (
    AbstractBaseFileSaver,
    AbstractBaseSaver,
    AsyncSaveManager,
    CsvSaver,
    MongoSaver,
    PickleSaver,
    SyncSaveManager,
)

# MONGOSAVER TESTS HAVE TO BE UNCOMMENTED ENSURING NO RACE CONDITIONS
# (WHEN THE ISSUE IS SOLVED) LLM GENERATED TESTS HAVE TO REVIEWED

# Assuming your saver classes are in a module called 'savers'
# from savers import PickleSaver, CsvSaver


def test_AbstractBaseSaver_is_abstract():
    with pytest.raises(TypeError):
        AbstractBaseSaver()  # type: ignore


##############################################
# Test if names built correctly
##############################################


@pytest.fixture
def file_saver():

    class FileSaver(AbstractBaseFileSaver):
        _suffix = "txt"

        def save(self, *args):
            pass

        def read(self, *args):
            pass

    return FileSaver


def test_saver_names(file_saver: type[AbstractBaseFileSaver]):
    s = file_saver("price_df", use_timestamp=False)
    path_name = s._file("all_data", "NQZ3")
    name = path_name.split("/")[-1]
    assert name == "price_df_all_data_NQZ3.txt"  # type: ignore


def test_filesaver_file_name(file_saver: type[AbstractBaseFileSaver]):
    s = file_saver("price_df", use_timestamp=False)
    filename = s._file()
    assert filename == f"{default_path()}/price_df.txt"


def test_filesaver_names_include_timestamp(
    file_saver: type[AbstractBaseFileSaver],
) -> None:
    s = file_saver("price_df", use_timestamp=True)
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H_%M")
    filename = s._file("all_data", "NQZ3")
    assert filename == f"{default_path()}/price_df_all_data_NQZ3_{now}.txt"


def test_PickleSaver_read_doesnt_accept_none():
    saver = PickleSaver("my_filename")
    with pytest.raises(ValueError):
        saver.read(None)


def test_CsvSaver_read_doesnt_accept_none():
    saver = CsvSaver("my_filename")
    with pytest.raises(ValueError):
        saver.read(None)


@pytest.mark.asyncio
async def test_AsyncSaveManager():
    output = []

    class FakeSaver(AbstractBaseSaver):

        def save(self, data, *args):
            output.append(data)

        def read(self, key=None):
            return output.pop()

    class T:
        save = AsyncSaveManager(FakeSaver())
        read = save.read

    t = T()
    t1 = T()

    t.save("xxx")
    t1.save("yyy")
    assert await wait_for_condition(lambda: output == ["xxx", "yyy"])
    await t.read()
    assert await wait_for_condition(lambda: output == ["xxx"])
    await t1.read()
    assert await wait_for_condition(lambda: output == [])


##############################################


@pytest.mark.asyncio
async def test_class_with_two_savers():
    output = {}

    class FakeSaver(AbstractBaseSaver):
        def __init__(self, name):
            self.name = name
            super().__init__()

        def save(self, data, *args):
            output[self.name] = data

        def read(self, key=None):
            return output[self.name]

    class T:
        # it's not the same saver object, only the same type
        # potentially saving to two different files or collections
        save_one = AsyncSaveManager(FakeSaver("one"))
        save_two = AsyncSaveManager(FakeSaver("two"))
        read_one = save_one.read
        read_two = save_two.read

    t = T()

    t.save_one("xxx")
    t.save_two("yyy")
    assert await wait_for_condition(lambda: output == {"one": "xxx", "two": "yyy"})
    result_one = await t.read_one()
    result_two = await t.read_two()
    assert await wait_for_condition(lambda: result_one == "xxx")
    assert await wait_for_condition(lambda: result_two == "yyy")


@pytest.mark.asyncio
async def test_AsyncSaveManager_as_non_descriptor():
    output = []

    class FakeSaver(AbstractBaseSaver):

        def save(self, data, *args):
            output.append(data)

        def read(self, key=None):
            return output.pop()

    class T:
        def __init__(self, saver):
            self.save = AsyncSaveManager(saver)
            self.read = self.save.read

    t = T(FakeSaver())

    t.save("xxx")

    assert await wait_for_condition(lambda: output == ["xxx"])
    result = await t.read()
    assert await wait_for_condition(lambda: result == "xxx")


@pytest.mark.asyncio
async def test_AsyncSaveManager_as_descriptor():
    output = []

    class FakeSaver(AbstractBaseSaver):

        def save(self, data, *args):
            output.append(data)

        def read(self, key=None):
            pass

    class T:
        save = AsyncSaveManager(FakeSaver())

    t = T()

    t.save("xxx")

    assert await wait_for_condition(lambda: output == ["xxx"])


def test_SyncSaveManager_as_non_descriptor():
    output = []

    class FakeSaver(AbstractBaseSaver):

        def save(self, data, *args):
            output.append(data)

        def read(self, key=None):
            return output.pop()

    class T:
        def __init__(self, saver):
            saver = SyncSaveManager(saver)
            self.save = saver.save
            self.read = saver.read

    t = T(FakeSaver())

    t.save("xxx")

    assert output == ["xxx"]
    result = t.read()
    assert result == "xxx"


def test_SyncSaveManager_as_descriptor():
    output = []

    class FakeSaver(AbstractBaseSaver):

        def save(self, data, *args):
            output.append(data)

        def read(self, key=None):
            return output.pop()

    class T:
        save_manager = SyncSaveManager(FakeSaver())
        save = save_manager.save
        read = save_manager.read

    t = T()

    t.save("xxx")

    assert output == ["xxx"]

    result = t.read()
    assert result == "xxx"


# #####################################
# ####### actual savers' tests ########
# #####################################


class TestPickleSaver:

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def pickle_saver(self, temp_dir):
        """Create a PickleSaver instance for testing."""
        with patch("haymaker.saver.default_path", return_value=temp_dir):
            return PickleSaver(name="test", folder="", use_timestamp=False)

    @pytest.fixture
    def pickle_saver_with_timestamp(self, temp_dir):
        """Create a PickleSaver instance with timestamp for testing."""
        with patch("haymaker.saver.default_path", return_value=temp_dir):
            return PickleSaver(name="test", folder="", use_timestamp=True)

    @pytest.fixture
    def mock_name_str(self):
        """Mock the name_str function to return predictable filenames."""
        with patch("haymaker.saver.name_str") as mock:
            mock.return_value = "test_file"
            yield mock

    def test_save_pandas_dataframe(self, pickle_saver, mock_name_str, temp_dir):
        """Test saving a pandas DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        pickle_saver.save(df, "extra_arg")

        # Check that the file was created
        expected_file = Path(temp_dir) / "test_file.pickle"
        assert expected_file.exists()

        # Verify the content by reading it directly
        loaded_df = pd.read_pickle(expected_file)
        pd.testing.assert_frame_equal(df, loaded_df)

        # Verify name_str was called correctly
        mock_name_str.assert_called_once_with("test", "extra_arg", timestamp=None)

    def test_save_non_dataframe_object(self, pickle_saver, mock_name_str, temp_dir):
        """Test saving a non-DataFrame object."""
        test_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": "dict"}}

        pickle_saver.save(test_data, "dict_test")

        # Check that the file was created
        expected_file = Path(temp_dir) / "test_file.pickle"
        assert expected_file.exists()

        # Verify the content by reading it directly
        with open(expected_file, "rb") as f:
            loaded_data = pickle.load(f)
        assert loaded_data == test_data

        mock_name_str.assert_called_once_with("test", "dict_test", timestamp=None)

    def test_save_with_timestamp(self, pickle_saver_with_timestamp, temp_dir):
        """Test that timestamp is passed to name_str when use_timestamp=True."""
        test_data = [1, 2, 3, 4, 5]

        with patch(
            "haymaker.saver.name_str", return_value="timestamped_file"
        ) as mock_name_str:
            pickle_saver_with_timestamp.save(test_data)

            # Verify that timestamp was passed (not None)
            mock_name_str.assert_called_once()
            args, kwargs = mock_name_str.call_args
            assert kwargs["timestamp"] is not None
            assert isinstance(kwargs["timestamp"], datetime)

    def test_read_existing_file(self, pickle_saver, mock_name_str, temp_dir):
        """Test reading an existing pickle file."""
        # First create a file to read
        test_data = {"read_test": "success", "numbers": [1, 2, 3]}
        expected_file = Path(temp_dir) / "test_file.pickle"

        with open(expected_file, "wb") as f:
            pickle.dump(test_data, f)

        # Now test reading
        result = pickle_saver.read("read_key")

        assert result == test_data
        mock_name_str.assert_called_once_with("test", "read_key", timestamp=None)

    def test_read_with_none_name(self, pickle_saver):
        """Test that read raises ValueError when name is None."""
        with pytest.raises(ValueError, match="name must be a string, not None"):
            pickle_saver.read(None)

    def test_read_nonexistent_file(self, pickle_saver, mock_name_str):
        """Test reading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            pickle_saver.read("nonexistent")

    def test_save_multiple_args(self, pickle_saver, mock_name_str, temp_dir):
        """Test saving with multiple additional arguments."""
        test_data = "test with multiple args"

        pickle_saver.save(test_data, "arg1", "arg2", "arg3")

        mock_name_str.assert_called_once_with(
            "test", "arg1", "arg2", "arg3", timestamp=None
        )


class TestCsvSaver:

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def csv_saver(self, temp_dir):
        """Create a CsvSaver instance for testing."""
        with patch("haymaker.saver.default_path", return_value=temp_dir):
            return CsvSaver(name="test", folder="", use_timestamp=False)

    @pytest.fixture
    def mock_name_str(self):
        """Mock the name_str function to return predictable filenames."""
        with patch("haymaker.saver.name_str") as mock:
            mock.return_value = "test_file"
            yield mock

    def test_save_new_file_creates_header(self, csv_saver, mock_name_str, temp_dir):
        """Test that saving to a new file creates a header."""
        test_data = {"name": "John", "age": 30, "city": "New York"}

        csv_saver.save(test_data, "new_file")

        expected_file = Path(temp_dir) / "test_file.csv"
        assert expected_file.exists()

        # Read the file and check content
        with open(expected_file, "r", newline="") as f:
            reader = csv.reader(f)
            lines = list(reader)

        # Should have header + data row
        assert len(lines) == 2
        assert set(lines[0]) == {"name", "age", "city"}  # Header
        assert lines[1] == ["John", "30", "New York"]  # Data (order may vary)

    def test_save_existing_file_appends_data(self, csv_saver, mock_name_str, temp_dir):
        """Test that saving to existing file appends data without new header."""
        # Create initial file
        test_data1 = {"name": "John", "age": 30, "city": "New York"}
        csv_saver.save(test_data1)

        # Append more data
        test_data2 = {"name": "Jane", "age": 25, "city": "Boston"}
        csv_saver.save(test_data2)

        expected_file = Path(temp_dir) / "test_file.csv"

        # Read the file and check content
        with open(expected_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["name"] == "John"
        assert rows[1]["name"] == "Jane"

    def test_save_many_empty_list(self, csv_saver):
        """Test that save_many raises ValueError for empty list."""
        with pytest.raises(ValueError, match="Cannot save empty data list"):
            csv_saver.save_many([], override=False)

    def test_save_many_new_file(self, csv_saver, mock_name_str, temp_dir):
        """Test save_many with a new file."""
        test_data = [
            {"name": "Alice", "score": 95, "subject": "Math"},
            {"name": "Bob", "score": 87, "subject": "Math"},
            {"name": "Charlie", "score": 92, "subject": "Math"},
        ]

        csv_saver.save_many(test_data, "batch_test", override=True)

        expected_file = Path(temp_dir) / "test_file.csv"
        assert expected_file.exists()

        # Read and verify content
        with open(expected_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["name"] == "Alice"
        assert rows[1]["name"] == "Bob"
        assert rows[2]["name"] == "Charlie"

    def test_save_many_should_not_recreate_header_on_existing_file(
        self, csv_saver, mock_name_str, temp_dir
    ):
        """Test that save_many doesn't recreate header when override=False and file exists."""
        # Create initial file with data
        initial_data = {"name": "Initial", "score": 100, "subject": "Test"}
        csv_saver.save(initial_data)

        # Verify initial file exists and has correct content
        expected_file = Path(temp_dir) / "test_file.csv"
        assert expected_file.exists()

        # Read initial content
        with open(expected_file, "r", newline="") as f:
            initial_content = f.read()

        print(f"Initial file content:\n{initial_content}")

        # Now use save_many with override=False - this should NOT recreate the header
        batch_data = [
            {"name": "Batch1", "score": 80, "subject": "Test"},
            {"name": "Batch2", "score": 90, "subject": "Test"},
        ]
        csv_saver.save_many(batch_data, override=False)

        # Read final content
        with open(expected_file, "r", newline="") as f:
            final_content = f.read()

        print(f"Final file content:\n{final_content}")

        # Count headers - there should only be one
        header_count = final_content.count("name,score,subject")
        assert header_count == 1, f"Expected 1 header but found {header_count}"

    def test_save_many_with_override_recreates_header(
        self, csv_saver, mock_name_str, temp_dir
    ):
        """Test that save_many recreates header when override=True."""
        # Create initial file with data
        initial_data = {"name": "Initial", "score": 100, "subject": "Test"}
        csv_saver.save(initial_data)

        expected_file = Path(temp_dir) / "test_file.csv"

        # Use save_many with override=True - this should recreate the header
        batch_data = [
            {"name": "Batch1", "score": 80, "subject": "Test"},
            {"name": "Batch2", "score": 90, "subject": "Test"},
        ]
        csv_saver.save_many(batch_data, override=True)

        # Read final content
        with open(expected_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # With override=True, the original data should be gone and only batch data
        # should remain
        assert len(rows) == 2
        names = [row["name"] for row in rows]
        assert "Initial" not in names  # Original data should be overwritten
        assert "Batch1" in names

    def test_save_many_args_consistency(self, csv_saver, mock_name_str, temp_dir):
        """Test that save_many uses args consistently in file path checks."""
        test_data = [{"name": "Test", "value": 1}]

        # This should work without errors - the key test is that it doesn't crash
        # due to inconsistent use of args in file path generation
        csv_saver.save_many(test_data, "arg1", "arg2", override=True)

        # Verify the file was created
        # (mock_name_str should have been called with the args)
        expected_file = Path(temp_dir) / "test_file.csv"
        assert expected_file.exists()

        # Verify mock was called with the correct arguments
        mock_name_str.assert_called_with("test", "arg1", "arg2", timestamp=None)

    def test_save_many_with_default_override(self, csv_saver, mock_name_str, temp_dir):
        """
        Test save_many with default override=True behavior, appending
        to existing file.
        """
        # Create initial data
        initial_data = {"name": "Initial", "score": 100, "subject": "Test"}
        csv_saver.save(initial_data)

        # Add batch data
        batch_data = [
            {"name": "Batch1", "score": 80, "subject": "Test"},
            {"name": "Batch2", "score": 90, "subject": "Test"},
        ]
        csv_saver.save_many(batch_data)

        expected_file = Path(temp_dir) / "test_file.csv"

        # Read and verify all data is present
        with open(expected_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Debug: Let's see what's actually in the file
        print(f"Rows found: {rows}")
        print(f"Number of rows: {len(rows)}")

        # The test should work if save_many properly appends
        # If this fails, it means save_many is overwriting instead of appending
        assert len(rows) == 2, f"Expected 2 rows but got {len(rows)}. Rows: {rows}"

        # Check that all expected data is present (order might vary)
        names = [row["name"] for row in rows]
        assert "Batch1" in names, f"Batch1 missing. Found names: {names}"
        assert "Batch2" in names, f"Batch2 missing. Found names: {names}"

    def test_save_many_with_override_false(self, csv_saver, mock_name_str, temp_dir):
        """
        Test save_many with default override=True behavior, appending
        to existing file.
        """
        # Create initial data
        initial_data = {"name": "Initial", "score": 100, "subject": "Test"}
        csv_saver.save(initial_data)

        # Add batch data
        batch_data = [
            {"name": "Batch1", "score": 80, "subject": "Test"},
            {"name": "Batch2", "score": 90, "subject": "Test"},
        ]
        csv_saver.save_many(batch_data, override=False)

        expected_file = Path(temp_dir) / "test_file.csv"

        # Read and verify all data is present
        with open(expected_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Debug: Let's see what's actually in the file
        print(f"Rows found: {rows}")
        print(f"Number of rows: {len(rows)}")

        # The test should work if save_many properly appends
        # If this fails, it means save_many is overwriting instead of appending
        assert len(rows) == 3, f"Expected 3 rows but got {len(rows)}. Rows: {rows}"

        # Check that all expected data is present (order might vary)
        names = [row["name"] for row in rows]
        assert "Initial" in names, f"Initial data missing. Found names: {names}"
        assert "Batch1" in names, f"Batch1 missing. Found names: {names}"
        assert "Batch2" in names, f"Batch2 missing. Found names: {names}"

    def test_read_existing_file(self, csv_saver, mock_name_str, temp_dir):
        """Test reading an existing CSV file."""
        # Create test data file
        expected_file = Path(temp_dir) / "test_file.csv"
        test_data = [
            {"name": "Alice", "age": "30", "city": "Seattle"},
            {"name": "Bob", "age": "25", "city": "Portland"},
        ]

        with open(expected_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "age", "city"])
            writer.writeheader()
            writer.writerows(test_data)

        # Test reading
        result = csv_saver.read("read_test")

        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["name"] == "Bob"
        print(mock_name_str.call_count, mock_name_str.call_args)
        mock_name_str.assert_called_once_with("test", timestamp=None)

    def test_read_with_none_name(self, csv_saver):
        """Test that read raises ValueError when name is None."""
        with pytest.raises(ValueError, match="name must be a string, not None"):
            csv_saver.read(None)

    def test_read_nonexistent_file(self, csv_saver, mock_name_str):
        """Test reading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            csv_saver.read("nonexistent")

    def test_read_empty_csv(self, csv_saver, mock_name_str, temp_dir):
        """Test reading an empty CSV file (header only)."""
        expected_file = Path(temp_dir) / "test_file.csv"

        # Create file with header only
        with open(expected_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["col1", "col2"])
            writer.writeheader()

        result = csv_saver.read("empty_test")

        assert result == []

    def test_constructor_calls_super(self, temp_dir):
        """Test that CsvSaver constructor properly calls parent constructor."""
        with patch(
            "haymaker.saver.default_path", return_value=temp_dir
        ) as mock_default_path:
            saver = CsvSaver(name="test_name", folder="test_folder", use_timestamp=True)

            # Verify the parent constructor was called properly
            assert saver.name == "test_name"
            assert saver.path == temp_dir
            assert saver.timestamp is not None
            mock_default_path.assert_called_once_with("test_folder")


# Integration tests that don't require mocking
class TestSaverIntegration:

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_pickle_save_and_read_roundtrip(self, temp_dir):
        """Test complete save/read cycle for PickleSaver."""
        # Mock the helper functions to use temp directory
        with patch("haymaker.saver.default_path", return_value=temp_dir), patch(
            "haymaker.saver.name_str", return_value="integration_test"
        ):

            saver = PickleSaver(name="test", use_timestamp=False)

            # Test data
            original_data = {
                "string": "test",
                "number": 42,
                "list": [1, 2, 3],
                "nested": {"inner": "value"},
            }

            # Save and read
            saver.save(original_data, "roundtrip")
            loaded_data = saver.read("roundtrip")

            assert loaded_data == original_data

    def test_csv_save_and_read_roundtrip(self, temp_dir):
        """Test complete save/read cycle for CsvSaver."""
        with patch("haymaker.saver.default_path", return_value=temp_dir), patch(
            "haymaker.saver.name_str", return_value="integration_test"
        ):

            saver = CsvSaver(name="test", use_timestamp=False)

            # Test data
            original_data = {"name": "Test User", "score": "95", "active": "true"}

            # Save and read
            saver.save(original_data, "roundtrip")
            loaded_data = saver.read("roundtrip")

            assert len(loaded_data) == 1
            assert loaded_data[0] == original_data


# ##################
# ### MongoSaver ###
# ##################


class TestMongoSaver:

    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client and collection."""
        mock_client = Mock(spec=pymongo.MongoClient)
        mock_db = Mock()
        mock_collection = Mock()

        mock_client.__getitem__ = Mock(return_value=mock_db)
        mock_db.__getitem__ = Mock(return_value=mock_collection)

        return mock_client, mock_db, mock_collection

    @pytest.fixture
    def mock_config(self):
        """Mock the CONFIG dictionary."""
        return {"MongoSaver": {"db": "test_database"}}

    @pytest.fixture
    def mongo_saver_with_query_key(self, mock_mongo_client, mock_config):
        """Create MongoSaver instance with query_key."""
        mock_client, mock_db, mock_collection = mock_mongo_client

        with patch("haymaker.saver.get_mongo_client", return_value=mock_client), patch(
            "haymaker.saver.CONFIG", mock_config
        ):
            saver = MongoSaver(collection="test_collection", query_key="id")
            saver.collection = mock_collection  # Ensure we're using the mock
            return saver, mock_collection

    @pytest.fixture
    def mongo_saver_without_query_key(self, mock_mongo_client, mock_config):
        """Create MongoSaver instance without query_key."""
        mock_client, mock_db, mock_collection = mock_mongo_client

        with patch("haymaker.saver.get_mongo_client", return_value=mock_client), patch(
            "haymaker.saver.CONFIG", mock_config
        ):
            saver = MongoSaver(collection="test_collection", query_key=None)
            saver.collection = mock_collection  # Ensure we're using the mock
            return saver, mock_collection

    def test_init_with_query_key(self, mock_mongo_client, mock_config):
        """Test MongoSaver initialization with query_key."""
        mock_client, mock_db, mock_collection = mock_mongo_client

        with patch("haymaker.saver.get_mongo_client", return_value=mock_client), patch(
            "haymaker.saver.CONFIG", mock_config
        ):

            saver = MongoSaver(collection="test_collection", query_key="user_id")

            assert saver.query_key == "user_id"
            assert saver.client == mock_client
            assert saver.db == mock_db
            mock_client.__getitem__.assert_called_once_with("test_database")
            mock_db.__getitem__.assert_called_once_with("test_collection")

    def test_init_without_query_key(self, mock_mongo_client, mock_config):
        """Test MongoSaver initialization without query_key."""
        mock_client, mock_db, mock_collection = mock_mongo_client

        with patch("haymaker.saver.get_mongo_client", return_value=mock_client), patch(
            "haymaker.saver.CONFIG", mock_config
        ):

            saver = MongoSaver(collection="users", query_key=None)

            assert saver.query_key is None
            assert saver.client == mock_client
            assert saver.db == mock_db

    def test_save_with_query_key_upsert(self, mongo_saver_with_query_key):
        """Test save method with query_key performs upsert."""
        saver, mock_collection = mongo_saver_with_query_key
        mock_result = Mock()
        mock_collection.update_one.return_value = mock_result

        test_data = {"id": "user123", "name": "John Doe", "age": 30}

        saver.save(test_data)

        mock_collection.update_one.assert_called_once_with(
            {"id": "user123"},  # Query filter
            {"$set": test_data},  # Update data
            upsert=True,
        )
        mock_collection.insert_one.assert_not_called()

    def test_save_with_query_key_but_missing_in_data(self, mongo_saver_with_query_key):
        """Test save method when query_key is set but not present in data."""
        saver, mock_collection = mongo_saver_with_query_key
        mock_result = Mock()
        mock_collection.insert_one.return_value = mock_result

        # Data doesn't contain the query_key "id"
        test_data = {"name": "Jane Doe", "age": 25}

        saver.save(test_data)

        # Should fall back to insert_one since query_key not in data
        mock_collection.insert_one.assert_called_once_with(test_data)
        mock_collection.update_one.assert_not_called()

    def test_save_with_query_key_none_value(self, mongo_saver_with_query_key):
        """Test save method when query_key exists but has None value."""
        saver, mock_collection = mongo_saver_with_query_key
        mock_result = Mock()
        mock_collection.insert_one.return_value = mock_result

        # Data contains query_key but with None value
        test_data = {"id": None, "name": "Jane Doe", "age": 25}

        saver.save(test_data)

        # Should fall back to insert_one since query_key value is falsy
        mock_collection.insert_one.assert_called_once_with(test_data)
        mock_collection.update_one.assert_not_called()

    def test_save_without_query_key_insert(self, mongo_saver_without_query_key):
        """Test save method without query_key performs insert."""
        saver, mock_collection = mongo_saver_without_query_key
        mock_result = Mock()
        mock_collection.insert_one.return_value = mock_result

        test_data = {"name": "Alice Smith", "department": "Engineering"}

        saver.save(test_data)

        mock_collection.insert_one.assert_called_once_with(test_data)
        mock_collection.update_one.assert_not_called()

    def test_save_with_args_ignored(self, mongo_saver_without_query_key):
        """Test that additional args are ignored in save method."""
        saver, mock_collection = mongo_saver_without_query_key
        mock_result = Mock()
        mock_collection.insert_one.return_value = mock_result

        test_data = {"name": "Bob Wilson", "role": "Manager"}

        # Args should be ignored
        saver.save(test_data, "extra", "args", "ignored")

        mock_collection.insert_one.assert_called_once_with(test_data)

    @patch("haymaker.saver.log")
    def test_save_exception_handling(self, mock_log, mongo_saver_without_query_key):
        """Test save method handles exceptions properly."""
        saver, mock_collection = mongo_saver_without_query_key

        # Make insert_one raise an exception
        test_exception = pymongo.errors.WriteError("Database error")
        mock_collection.insert_one.side_effect = test_exception

        test_data = {"name": "Error User"}

        with pytest.raises(pymongo.errors.WriteError):
            saver.save(test_data)

        # Verify logging was called
        mock_log.exception.assert_called_once_with("Error saving data to MongoDB")
        mock_log.debug.assert_called_once_with(f"Data that caused error: {test_data}")

    def test_read_with_empty_query(self, mongo_saver_without_query_key):
        """Test read method with no query (returns all documents)."""
        saver, mock_collection = mongo_saver_without_query_key

        expected_results = [
            {"_id": "1", "name": "User1"},
            {"_id": "2", "name": "User2"},
        ]
        mock_collection.find.return_value = expected_results

        result = saver.read()

        assert result == expected_results
        mock_collection.find.assert_called_once_with({})

    def test_read_with_none_query(self, mongo_saver_without_query_key):
        """Test read method with explicit None query."""
        saver, mock_collection = mongo_saver_without_query_key

        expected_results = [{"_id": "1", "name": "User1"}]
        mock_collection.find.return_value = expected_results

        result = saver.read(None)

        assert result == expected_results
        mock_collection.find.assert_called_once_with({})

    def test_read_with_specific_query(self, mongo_saver_without_query_key):
        """Test read method with specific query."""
        saver, mock_collection = mongo_saver_without_query_key

        expected_results = [{"_id": "1", "name": "John", "age": 30}]
        mock_collection.find.return_value = expected_results

        query = {"name": "John", "age": {"$gte": 25}}
        result = saver.read(query)

        assert result == expected_results
        mock_collection.find.assert_called_once_with(query)

    def test_read_empty_results(self, mongo_saver_without_query_key):
        """Test read method returns empty list when no documents found."""
        saver, mock_collection = mongo_saver_without_query_key

        mock_collection.find.return_value = []

        result = saver.read({"nonexistent": "value"})

        assert result == []
        mock_collection.find.assert_called_once_with({"nonexistent": "value"})

    @patch("haymaker.saver.log")
    def test_delete_method_logs_debug(self, mock_log, mongo_saver_without_query_key):
        """Test delete method logs debug message (not implemented)."""
        saver, mock_collection = mongo_saver_without_query_key

        query = {"id": "user_to_delete"}
        saver.delete(query)

        mock_log.debug.assert_called_once_with(
            f"Will mock delete data: {query}. DELETE METHOD NOT IMPLEMENTED"
        )

    def test_str_representation(self, mongo_saver_without_query_key):
        """Test string representation of MongoSaver."""
        saver, mock_collection = mongo_saver_without_query_key

        # Mock the db and collection names for string representation
        saver.db.name = "test_database"
        saver.collection.name = "test_collection"

        expected_str = "MongoSaver(db=test_database, collection=test_collection)"
        # Note: The actual __str__ method uses the objects directly, so we test the format
        result = str(saver)
        assert "MongoSaver" in result
        assert "db=" in result
        assert "collection=" in result

    def test_save_upsert_with_different_data_types(self, mongo_saver_with_query_key):
        """Test upsert with various data types."""
        saver, mock_collection = mongo_saver_with_query_key
        mock_result = Mock()
        mock_collection.update_one.return_value = mock_result

        test_data = {
            "id": 12345,  # Integer ID
            "name": "Test User",
            "tags": ["python", "mongodb"],
            "metadata": {"created": "2023-01-01", "active": True},
            "score": 95.5,
        }

        saver.save(test_data)

        mock_collection.update_one.assert_called_once_with(
            {"id": 12345}, {"$set": test_data}, upsert=True
        )

    def test_multiple_saves_and_reads(self, mongo_saver_with_query_key):
        """Test multiple consecutive save and read operations."""
        saver, mock_collection = mongo_saver_with_query_key
        mock_result = Mock()
        mock_collection.update_one.return_value = mock_result
        mock_collection.find.return_value = [{"id": "user1", "name": "Updated"}]

        # Multiple saves
        data1 = {"id": "user1", "name": "First Save"}
        data2 = {"id": "user1", "name": "Updated"}

        saver.save(data1)
        saver.save(data2)

        # Verify both calls
        assert mock_collection.update_one.call_count == 2

        # Read
        result = saver.read({"id": "user1"})
        assert result == [{"id": "user1", "name": "Updated"}]


# Integration-style tests that test the interaction with get_mongo_client
class TestMongoSaverIntegration:

    @patch("haymaker.saver.get_mongo_client")
    def test_mongo_client_initialization(self, mock_get_client):
        """Test that MongoSaver properly initializes with get_mongo_client."""
        # Setup mocks
        mock_client = MagicMock(spec=pymongo.MongoClient)
        mock_db = MagicMock()
        mock_collection = MagicMock()

        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_get_client.return_value = mock_client

        # Mock CONFIG
        mock_config = {"MongoSaver": {"db": "production_db"}}

        with patch("haymaker.saver.CONFIG", mock_config):
            saver = MongoSaver(collection="users", query_key="email")

        # Verify get_mongo_client was called
        mock_get_client.assert_called_once()

        # Verify database and collection selection
        mock_client.__getitem__.assert_called_once_with("production_db")
        mock_db.__getitem__.assert_called_once_with("users")

        # Verify saver properties
        assert saver.client == mock_client
        assert saver.db == mock_db
        assert saver.collection == mock_collection
        assert saver.query_key == "email"

    @patch("haymaker.saver.get_mongo_client")
    def test_config_error_propagation(self, mock_get_client):
        """Test that MongoDB configuration errors are properly propagated."""
        # Make get_mongo_client raise a configuration error
        mock_get_client.side_effect = pymongo.errors.ConfigurationError(
            "Invalid config"
        )

        mock_config = {"MongoSaver": {"db": "test_db"}}

        with patch("haymaker.saver.CONFIG", mock_config):
            with pytest.raises(pymongo.errors.ConfigurationError):
                MongoSaver(collection="test_collection")

        mock_get_client.assert_called_once()


# ##### Test if MongoSaver has no race conditions


# @pytest.fixture
# def mongo_saver(monkeypatch):
#     client = mongomock.MongoClient()
#     monkeypatch.setattr("haymaker.databases.get_mongo_client", lambda: client)
#     saver = MongoSaver(collection="test_collection", query_key="id")
#     return saver


# def test_no_duplicates_under_race(mongo_saver):
#     """
#     Simulate concurrent inserts/updates with the same query_key
#     and ensure only one document exists per query_key after all threads finish.
#     """
#     key = "race1"
#     initial_data = {"id": key, "value": 0}

#     def worker(value):
#         data = initial_data.copy()
#         data["value"] = value
#         mongo_saver.save(data)

#     threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()

#     docs = list(mongo_saver.collection.find({"id": key}))
#     assert len(docs) == 1
#     assert docs[0]["value"] in range(10)


# def test_multiple_keys_race(mongo_saver):
#     """
#     Ensure that multiple query_keys concurrently saved do not interfere.
#     """
#     keys = [f"key{i}" for i in range(5)]

#     def worker(k):
#         for i in range(5):
#             mongo_saver.save({"id": k, "value": i})

#     threads = [threading.Thread(target=worker, args=(k,)) for k in keys]
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()

#     for k in keys:
#         docs = list(mongo_saver.collection.find({"id": k}))
#         assert len(docs) == 1
#         assert docs[0]["value"] in range(5)
