def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    given_month, given_day = date

    # Convert (month, day) into day-of-year (simple 365-day year)
    def day_of_year(month, day):
        days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
        return sum(days_in_month[:month-1]) + day

    given_doy = day_of_year(given_month, given_day)

    next_doy = None
    birthday = None

    for bday in birthdays:
        bday_doy = day_of_year(bday[0], bday[1])

        # Birthday after given date
        if bday_doy > given_doy:
            if next_doy is None or bday_doy < next_doy:
                next_doy = bday_doy
                birthday = bday

    # If no future birthday found, wrap to next year (earliest birthday)
    if birthday is None:
        birthday = min(birthdays.keys(), key=lambda d: day_of_year(d[0], d[1]))

    list_of_names = birthdays[birthday]
    return birthday, list_of_names
