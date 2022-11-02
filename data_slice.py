import glob
import pandas


def get_data(path, frac):
    """Read in the data"""
    filenames = glob.glob(path)
    df = pandas.concat([sample(f, frac) for f in filenames], ignore_index=True)
    return df


def sample(path, frac):
    """select a fraction of the data"""
    print(f"Read {path}")
    df = pandas.read_csv(path)
    return df.sample(frac=frac)


def main():
    """Create a random sample of all the data"""
    # get 10% of the actual data
    df = get_data("data/2018*-citibike-tripdata.csv.zip", 0.1)
    # write back
    df.to_csv("data/sample.csv", index=False)


if __name__ == '__main__':
    main()
