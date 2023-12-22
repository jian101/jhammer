from jhammer.type_conversion import convert_2_dtype


def aggregate_mean_std(mean1, std1, num1, mean2, std2, num2, ddof=1):
    """
    Calculate the mean and std of a group of data composed by two groups of data,
    given the mean, the std, and the number of data of the two subsets.

    Args:
          mean1: mean value of group1
          std1: std value of group1
          num1: number of data in group1
          mean2: mean value of group2
          std2: std value of group2
          num2: number of data in group2
          ddof: means of delta degrees of freedom. Default is 1 for unbiased estimation.
    """
    assert num1 != 0 and num2 != 0, f"num1 and num2 can not be zero. num1: {num1}, num2: {num2}"
    mean1 = convert_2_dtype(mean1, float)
    std1 = convert_2_dtype(std1, float)
    mean2 = convert_2_dtype(mean2, float)
    std2 = convert_2_dtype(std2, float)

    mean = (num1 * mean1 + num2 * mean2) / (num1 + num2)
    d = ((num1 - ddof) * std1 ** 2 + (num2 - ddof) * std2 ** 2) / (num1 + num2 - ddof) + \
        num1 * num2 * (mean1 - mean2) ** 2 / (num1 + num2) / (num1 + num2 - ddof)
    std = d ** 0.5
    return mean, std
