import pandas as pd
from bs4 import BeautifulSoup

with open('Raw Data/Website.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')

columns = [
    "TEAM", "G", "W", "SEED", "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D", "TOR", "TORD",
    "ORB", "DRB", "FTR", "FTRD", "2P_O", "2P_D", "3P_O", "3P_D",
    "ADJ_T", "WAB"
]

data_rows = []

for row in soup.select('tr.seedrow'):
    cells = row.find_all('td')

    team_cell = cells[1]
    team_link = team_cell.find('a')
    
    seed_span = team_cell.find('span', class_='lowrow')

    team_name = team_link.contents[0].strip()
    seed = seed_span.get_text(strip=True).split(' seed')[0]

    if not seed:
        continue
    
    record = cells[4].get_text(strip=True)
    wins, losses = map(int, record.split('-'))
    games = wins + losses

    team_data = [
        team_name,                      # TEAM
        games,                          # G
        wins,                           # W
        seed,                           # SEED
        cells[5].get_text(strip=True),  # ADJOE
        cells[6].get_text(strip=True),  # ADJDE
        cells[7].get_text(strip=True),  # BARTHAG
        cells[8].get_text(strip=True),  # EFG_O
        cells[9].get_text(strip=True),  # EFG_D
        cells[10].get_text(strip=True), # TOR
        cells[11].get_text(strip=True), # TORD
        cells[12].get_text(strip=True), # ORB
        cells[13].get_text(strip=True), # ORB_D
        cells[14].get_text(strip=True), # FTR
        cells[15].get_text(strip=True), # FTRD
        cells[16].get_text(strip=True), # 2P_O
        cells[17].get_text(strip=True), # 2P_D
        cells[18].get_text(strip=True), # 3P_O
        cells[19].get_text(strip=True), # 3P_D
        cells[22].get_text(strip=True), # ADJ_T
        cells[23].get_text(strip=True)  # WAB
    ]
    data_rows.append(team_data)

df = pd.DataFrame(data_rows, columns=columns)
df['WAB'] = df['WAB'].str.replace('+', '', regex=False).astype(float)

df.to_csv('Transformed Data/2025 Stats.csv', index=False)