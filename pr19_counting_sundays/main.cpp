#include <iostream>
#include <map>

int get_all_sundays() {
    int number_sundays = 0;

    // we zero-index both
    int currentYear = 1900;
    int currentMonth = 0;
    int currentDay = 0;
    int daysUntilNextSunday = 6;
    int daysInCurrentMonth = 31;

    while (currentYear < 2001) {
        currentDay += daysUntilNextSunday;
        daysUntilNextSunday = 7;

        if (currentDay >= daysInCurrentMonth) {
            currentDay = currentDay % daysInCurrentMonth;
            if (currentDay == 0 && currentYear > 1900) {
                number_sundays++;
            }
            currentMonth++;
            if (currentMonth == 12) {
                currentYear++;
                currentMonth = 0;
            }

            daysInCurrentMonth = 31;
            if (currentMonth == 8 || currentMonth == 3 || currentMonth == 5 || currentMonth == 10) {
                daysInCurrentMonth = 30;
            }
            if (currentMonth == 1) {
                daysInCurrentMonth = 28;
                if (currentYear % 4 == 0 && (currentYear % 400 == 0 || currentYear % 100 != 0)) {
                    daysInCurrentMonth = 29;
                }
            }
        }
    }

    return number_sundays;
}

int main() {
    std::cout << get_all_sundays() << std::endl;
    return 0;
}
