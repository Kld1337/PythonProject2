Specyfikacja aplikacji Bioassay Data Hub
1. Nazwa aplikacji
Bioassay Data Hub

2. Wybrany temat projektu
Analiza danych pomiarowych z urządzeń

Aplikacja analizuje dane generowane przez czytnik mikropłytek. Dane wejściowe są eksportowane z urządzenia do plików XLSX i zawierają wyniki pomiarów płytek 96-dołkowych.

3. Cel aplikacji
Celem aplikacji jest automatyzacja przetwarzania danych pomiarowych pochodzących z czytnika mikropłytek.

Aplikacja ma ograniczać konieczność ręcznego przepisywania wyników, wykonywania powtarzalnych obliczeń oraz ręcznego przygotowywania wykresów.

System powinien umożliwiać:

wczytanie pliku pomiarowego,
wykrycie rodzaju pomiaru,
odczyt mapy próbek,
wykonanie analizy statystycznej,
przeprowadzenie analizy właściwej dla wybranego testu,
przygotowanie wykresów,
wygenerowanie raportu,
zapis eksperymentu w lokalnej bazie danych,
ponowne otwarcie i analizę zapisanego eksperymentu.
4. Użytkownik docelowy
Aplikacja jest przeznaczona dla osób analizujących dane biologiczne i laboratoryjne uzyskane z czytnika mikropłytek.

Użytkownik powinien znać:

układ płytki 96-dołkowej,
znaczenie identyfikatorów próbek,
podstawowe pojęcia statystyczne,
zasady wykonywania analiz biologicznych,
współczynniki rozcieńczeń i kryteria testów.
5. Dane wejściowe
5.1. Format danych
Podstawowym formatem danych wejściowych jest:

XLSX
Plik może zawierać jeden lub więcej arkuszy.

5.2. Obsługiwane bloki pomiarowe
Aplikacja powinna automatycznie rozpoznawać:

Abs
RFU
RLU
Sample
Znaczenie bloków:

Abs – absorbancja,
RFU – fluorescencja,
RLU – luminescencja,
Sample – mapa identyfikatorów próbek.
5.3. Układ płytki
Aplikacja obsługuje płytki 96-dołkowe:

8 wierszy: A–H
12 kolumn: 1–12
Każdy dołek jest identyfikowany przez połączenie oznaczenia wiersza oraz numeru kolumny, np.:

A1
B5
H12
6. Interfejs użytkownika
Interfejs zostanie wykonany w bibliotece Streamlit.

Aplikacja powinna działać w przeglądarce internetowej i posiadać szeroki układ ekranu.

7. Nawigacja główna
W panelu bocznym użytkownik może wybrać jeden z dwóch głównych modułów:

Import i analiza pliku
Baza eksperymentów
7.1. Import i analiza pliku
Moduł służy do:

wczytania pliku XLSX,
wyboru pliku,
wyboru arkusza,
wykrycia bloków pomiarowych,
uruchomienia analizy,
zapisania eksperymentu.
7.2. Baza eksperymentów
Moduł służy do:

wyboru projektu,
wyboru eksperymentu,
przeglądania metadanych,
przeglądania pomiarów,
ponownego uruchamiania analizy,
porównywania eksperymentów,
eksportowania danych,
usuwania eksperymentów.
8. Zakładki importu
8.1. Arkusz
Zakładka powinna pokazywać:

nazwę pliku,
nazwę arkusza,
liczbę wierszy i kolumn,
pełną zawartość arkusza,
listę automatycznie wykrytych bloków.
8.2. Pomiar
Zakładka powinna umożliwiać:

wybór bloku pomiarowego,
wyświetlenie rodzaju pomiaru,
wyświetlenie jednostki,
wyświetlenie zakresu wyników,
wyświetlenie tabeli 96 dołków,
wyświetlenie heatmapy,
eksport wyników do CSV.
8.3. Mapa
Zakładka powinna:

odczytać sekcję Sample,
przypisać identyfikatory do dołków,
wyświetlić liczbę opisanych dołków,
wyświetlić liczbę unikalnych identyfikatorów,
pokazać mapę płytki,
umożliwić eksport mapy do CSV.
8.4. Dane
Zakładka powinna połączyć pomiary z mapą próbek przy użyciu oznaczenia dołka.

Wynikowa tabela powinna zawierać co najmniej:

well
plate_row
plate_column
raw_value
sample_id
sample_status
8.5. Statystyka
Zakładka powinna umożliwiać ustawienie:

oczekiwanej liczby powtórzeń,
limitu CV.
Powinna obliczać:

liczebność,
średnią,
odchylenie standardowe,
CV,
minimum,
maksimum,
status kontroli jakości.
9. Moduły analityczne
9.1. Żywotność
Moduł powinien:

rozpoznawać blank,
rozpoznawać próbki i kontrole,
wykonywać korekcję sygnału,
obliczać żywotność procentową,
obliczać średnią, SD i CV,
generować wykres dawka–odpowiedź,
wykonywać dopasowanie 4PL,
obliczać IC50,
obliczać R²,
generować podsumowanie wyników.
9.2. Hemoliza
Moduł powinien obsługiwać:

Kwalifikację krwi PFH/TBH
Właściwy test hemolizy
Analiza powinna obejmować:

krzywą kalibracyjną,
kalibratory,
kontrole QC,
PFH,
TBH lub TBHd,
kontrolę interferencji,
NC i PC,
procent hemolizy,
ocenę poprawności eksperymentu,
wykres wyników,
krótkie wnioski.
9.3. BCA
Moduł powinien:

wykorzystywać blok absorbancji,
rozpoznawać standardy i blank,
wykonywać korekcję blanku,
wykonywać regresję liniową,
obliczać R²,
obliczać stężenia białka,
uwzględniać współczynnik rozcieńczenia,
obliczać SD i CV,
generować wykres standardów i próbek.
Dodatkowo moduł powinien zawierać kalkulator Western blot.

Kalkulator powinien obliczać:

wymaganą objętość próbki,
objętość buforu Laemmli 2×,
objętość wody,
całkowitą ilość białka,
końcową objętość mieszaniny.
9.4. Stężenie glukozy
Moduł powinien:

rozpoznawać blank, standardy, QC i próbki,
wykonywać regresję liniową,
obliczać zgodność wartości obliczonej z nominalną,
obliczać średnią, SD oraz CV,
analizować powtórzenia techniczne,
analizować niezależne przygotowania,
uwzględniać współczynnik rozcieńczenia,
obliczać końcowy wynik w g/L,
wyświetlać zakres specyfikacji,
umożliwiać zmianę granic specyfikacji,
umożliwiać ręczne wykluczanie wyników,
opcjonalnie obliczać recovery próbek wzbogaconych.
10. Wykluczanie wyników
Aplikacja powinna umożliwiać ręczne wykluczenie pojedynczego dołka.

Wykluczony wynik:

pozostaje widoczny w danych surowych,
nie uczestniczy w obliczeniach,
jest przedstawiony w osobnej tabeli,
może mieć podany powód wykluczenia.
Po wykluczeniu aplikacja powinna ponownie obliczać:

średnią,
SD,
CV,
stężenie,
wynik końcowy.
11. Obliczenia statystyczne
Aplikacja wykorzystuje biblioteki Pandas i NumPy.

Podstawowe obliczenia:

średnia
mediana
odchylenie standardowe
minimum
maksimum
liczba wyników
CV
regresja liniowa
R²
model 4PL
IC50
Współczynnik zmienności jest obliczany jako:

CV [%] = SD / średnia × 100
12. Wizualizacje
Aplikacja powinna generować:

heatmapy płytek,
wykresy słupkowe,
wykresy punktowe,
wykresy z SD,
wykresy dawka–odpowiedź,
krzywe kalibracyjne,
wykresy porównawcze,
wykresy z zakresem specyfikacji.
Wykresy są tworzone przy użyciu biblioteki Plotly.

13. Raportowanie
Aplikacja powinna umożliwiać eksport:

CSV
XLSX
Raport może zawierać:

dane surowe,
mapę próbek,
połączone dane,
statystyki,
wyniki analizy,
parametry dopasowania,
przygotowanie próbek,
listę wyników wykluczonych.
14. Baza danych
Aplikacja wykorzystuje lokalną bazę SQLite.

14.1. Tabela experiments
Tabela przechowuje:

identyfikator eksperymentu,
nazwę projektu,
rodzaj analizy,
datę eksperymentu,
typ pomiaru,
jednostkę,
nazwę pliku,
nazwę arkusza,
operatora,
punkt czasowy,
uwagi.
14.2. Tabela measurements
Tabela przechowuje:

identyfikator eksperymentu,
oznaczenie dołka,
wiersz płytki,
kolumnę płytki,
identyfikator próbki,
wartość pomiarową,
status przypisania próbki.
Każdy eksperyment powinien zawierać 96 rekordów pomiarowych.

15. Usuwanie eksperymentów
Usunięcie eksperymentu wymaga:

wyboru eksperymentu,
zaznaczenia potwierdzenia,
wpisania dokładnego identyfikatora,
naciśnięcia przycisku usuwania.
Usunięcie eksperymentu powoduje również usunięcie wszystkich powiązanych pomiarów.

16. Obsługa błędów
Aplikacja powinna wyświetlać czytelne komunikaty w przypadku:

braku bloku pomiarowego,
braku sekcji Sample,
niepełnej płytki,
braku wymaganych standardów,
braku blanku,
braku kontroli,
nieprawidłowej liczby pomiarów,
próby zapisania istniejącego identyfikatora,
wartości poza zakresem krzywej,
braku danych do wykonania analizy.
17. Wymagania niefunkcjonalne
Aplikacja powinna:

działać lokalnie,
posiadać czytelny interfejs,
nie wymagać połączenia z internetem po instalacji zależności,
zachowywać dane w lokalnej bazie,
działać w systemie Windows,
umożliwiać odtworzenie środowiska z requirements.txt,
nie publikować lokalnej bazy danych w repozytorium.
18. Wykorzystanie AI
Narzędzia AI mogą wspierać:

projektowanie interfejsu,
przygotowanie fragmentów kodu,
refaktoryzację,
wyszukiwanie błędów,
tworzenie testów,
przygotowanie dokumentacji.
Autor pozostaje odpowiedzialny za:

poprawność założeń,
znajomość zastosowanych wzorów,
testowanie kodu,
interpretację wyników,
ochronę danych.
19. Planowane rozszerzenia
Planowane dalsze funkcje:

pełna walidacja modułu RiboGreen,
analiza GFP mRNA,
analiza Fluc mRNA,
trwałe zapisywanie wykluczeń w bazie,
dodatkowe porównania eksperymentów,
dodatkowe formaty raportów,
dalsza rozbudowa metadanych eksperymentów.
20. Kryteria ukończenia projektu
Projekt uznaje się za gotowy do prezentacji, jeżeli:

aplikacja uruchamia się bez błędu,
wczytuje przykładowy plik XLSX,
pokazuje dane i mapę płytki,
wykonuje obliczenia statystyczne,
generuje wykres,
umożliwia eksport wyników,
pokazuje podsumowanie,
zapisuje eksperyment do SQLite,
pozwala ponownie otworzyć zapisany eksperyment,
zawiera README,
zawiera specyfikację,
zawiera przykładowe dane,
zawiera zrzuty ekranu.