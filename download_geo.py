'''
    Getting geographic information from the lat/lon coordinates
    using google maps API

    This script only needs to be run when the output file isn't already saved


    NOT NEEDED - FILE CAN BE DELETED
'''

import credentials
import googlemaps
import pandas as pd
import numpy as np
import os

## All countries and their codes w/ associated continent
continent = { # see https://gist.github.com/nobuti/3816985
"AF":"Asia", # "Islamic Republic of Afghanistan")
"AX":"Europe", # "Åland Islands")
"AL":"Europe", # "Republic of Albania")
"DZ":"Africa", # "People's Democratic Republic of Algeria")
"AS":"Oceania", # "American Samoa")
"AD":"Europe", # "Principality of Andorra")
"AO":"Africa", # "Republic of Angola")
"AI":"North America", # "Anguilla")
"AQ":"Antarctica", # "Antarctica (the territory South of 60 deg S)")
"AG":"North America", # "Antigua and Barbuda")
"AR":"South America", # "Argentine Republic")
"AM":"Asia", # "Republic of Armenia")
"AW":"North America", # "Aruba")
"AU":"Oceania", # "Commonwealth of Australia")
"AT":"Europe", # "Republic of Austria")
"AZ":"Asia", # "Republic of Azerbaijan")
"BS":"North America", # "Commonwealth of the Bahamas")
"BH":"Asia", # "Kingdom of Bahrain")
"BD":"Asia", # "People's Republic of Bangladesh")
"BB":"North America", # "Barbados")
"BY":"Europe", # "Republic of Belarus")
"BE":"Europe", # "Kingdom of Belgium")
"BZ":"North America", # "Belize")
"BJ":"Africa", # "Republic of Benin")
"BM":"North America", # "Bermuda")
"BT":"Asia", # "Kingdom of Bhutan")
"BO":"South America", # "Plurinational State of Bolivia")
"BQ":"North America", # '535'
"BA":"Europe", # "Bosnia and Herzegovina")
"BW":"Africa", # "Republic of Botswana")
"BV":"Antarctica", # "Bouvet Island (Bouvetoya)")
"BR":"South America", # "Federative Republic of Brazil")
"IO":"Asia", # "British Indian Ocean Territory (Chagos Archipelago)")
"VG":"North America", # "British Virgin Islands")
"BN":"Asia", # "Brunei Darussalam")
"BG":"Europe", # "Republic of Bulgaria")
"BF":"Africa", # "Burkina Faso")
"BI":"Africa", # "Republic of Burundi")
"KH":"Asia", # "Kingdom of Cambodia")
"CM":"Africa", # "Republic of Cameroon")
"CA":"North America", # "Canada")
"CV":"Africa", # "Republic of Cape Verde")
"KY":"North America", # "Cayman Islands")
"CF":"Africa", # "Central African Republic")
"TD":"Africa", # "Republic of Chad")
"CL":"South America", # "Republic of Chile")
"CN":"Asia", # "People's Republic of China")
"CX":"Asia", # "Christmas Island")
"CC":"Asia", # "Cocos (Keeling) Islands")
"CO":"South America", # "Republic of Colombia")
"KM":"Africa", # "Union of the Comoros")
"CD":"Africa", # "Democratic Republic of the Congo")
"CG":"Africa", # "Republic of the Congo")
"CK":"Oceania", # "Cook Islands")
"CR":"North America", # "Republic of Costa Rica")
"CI":"Africa", # "Republic of Cote d'Ivoire")
"HR":"Europe", # "Republic of Croatia")
"CU":"North America", # "Republic of Cuba")
"CW":"North America", # "Curaçao")
"CY":"Asia", # "Republic of Cyprus")
"CZ":"Europe", # "Czech Republic")
"DK":"Europe", # "Kingdom of Denmark")
"DJ":"Africa", # "Republic of Djibouti")
"DM":"North America", # "Commonwealth of Dominica")
"DO":"North America", # "Dominican Republic")
"EC":"South America", # "Republic of Ecuador")
"EG":"Africa", # "Arab Republic of Egypt")
"SV":"North America", # "Republic of El Salvador")
"GQ":"Africa", # "Republic of Equatorial Guinea")
"ER":"Africa", # "State of Eritrea")
"EE":"Europe", # "Republic of Estonia")
"ET":"Africa", # "Federal Democratic Republic of Ethiopia")
"FO":"Europe", # "Faroe Islands")
"FK":"South America", # "Falkland Islands (Malvinas)")
"FJ":"Oceania", # "Republic of Fiji")
"FI":"Europe", # "Republic of Finland")
"FR":"Europe", # "French Republic")
"GF":"South America", # "French Guiana")
"PF":"Oceania", # "French Polynesia")
"TF":"Antarctica", # "French Southern Territories")
"GA":"Africa", # "Gabonese Republic")
"GM":"Africa", # "Republic of the Gambia")
"GE":"Asia", # "Georgia")
"DE":"Europe", # "Federal Republic of Germany")
"GH":"Africa", # "Republic of Ghana")
"GI":"Europe", # "Gibraltar")
"GR":"Europe", # "Hellenic Republic Greece")
"GL":"North America", # "Greenland")
"GD":"North America", # "Grenada")
"GP":"North America", # "Guadeloupe")
"GU":"Oceania", # "Guam")
"GT":"North America", # "Republic of Guatemala")
"GG":"Europe", # "Bailiwick of Guernsey")
"GN":"Africa", # "Republic of Guinea")
"GW":"Africa", # "Republic of Guinea-Bissau")
"GY":"South America", # "Co-operative Republic of Guyana")
"HT":"North America", # "Republic of Haiti")
"HM":"Antarctica", # "Heard Island and McDonald Islands")
"VA":"Europe", # "Holy See (Vatican City State)")
"HN":"North America", # "Republic of Honduras")
"HK":"Asia", # "Hong Kong Special Administrative Region of China")
"HU":"Europe", # "Hungary")
"IS":"Europe", # "Republic of Iceland")
"IN":"Asia", # "Republic of India")
"ID":"Asia", # "Republic of Indonesia")
"IR":"Asia", # "Islamic Republic of Iran")
"IQ":"Asia", # "Republic of Iraq")
"IE":"Europe", # "Ireland")
"IM":"Europe", # "Isle of Man")
"IL":"Asia", # "State of Israel")
"IT":"Europe", # "Italian Republic")
"JM":"North America", # "Jamaica")
"JP":"Asia", # "Japan")
"JE":"Europe", # "Bailiwick of Jersey")
"JO":"Asia", # "Hashemite Kingdom of Jordan")
"KZ":"Asia", # "Republic of Kazakhstan")
"KE":"Africa", # "Republic of Kenya")
"KI":"Oceania", # "Republic of Kiribati")
"KP":"Asia", # "Democratic People's Republic of Korea")
"KR":"Asia", # "Republic of Korea")
"KW":"Asia", # "State of Kuwait")
"KG":"Asia", # "Kyrgyz Republic")
"LA":"Asia", # "Lao People's Democratic Republic")
"LV":"Europe", # "Republic of Latvia")
"LB":"Asia", # "Lebanese Republic")
"LS":"Africa", # "Kingdom of Lesotho")
"LR":"Africa", # "Republic of Liberia")
"LY":"Africa", # "Libya")
"LI":"Europe", # "Principality of Liechtenstein")
"LT":"Europe", # "Republic of Lithuania")
"LU":"Europe", # "Grand Duchy of Luxembourg")
"MO":"Asia", # "Macao Special Administrative Region of China")
"MK":"Europe", # "Republic of Macedonia")
"MG":"Africa", # "Republic of Madagascar")
"MW":"Africa", # "Republic of Malawi")
"MY":"Asia", # "Malaysia")
"MV":"Asia", # "Republic of Maldives")
"ML":"Africa", # "Republic of Mali")
"MT":"Europe", # "Republic of Malta")
"MH":"Oceania", # "Republic of the Marshall Islands")
"MQ":"North America", # "Martinique")
"MR":"Africa", # "Islamic Republic of Mauritania")
"MU":"Africa", # "Republic of Mauritius")
"YT":"Africa", # "Mayotte")
"MX":"North America", # "United Mexican States")
"FM":"Oceania", # "Federated States of Micronesia")
"MD":"Europe", # "Republic of Moldova")
"MC":"Europe", # "Principality of Monaco")
"MN":"Asia", # "Mongolia")
"ME":"Europe", # "Montenegro")
"MS":"North America", # "Montserrat")
"MA":"Africa", # "Kingdom of Morocco")
"MZ":"Africa", # "Republic of Mozambique")
"MM":"Asia", # "Republic of the Union of Myanmar")
"NA":"Africa", # "Republic of Namibia")
"NR":"Oceania", # "Republic of Nauru")
"NP":"Asia", # "Federal Democratic Republic of Nepal")
"NL":"Europe", # "Kingdom of the Netherlands")
"NC":"Oceania", # "New Caledonia")
"NZ":"Oceania", # "New Zealand")
"NI":"North America", # "Republic of Nicaragua")
"NE":"Africa", # "Republic of Niger")
"NG":"Africa", # "Federal Republic of Nigeria")
"NU":"Oceania", # "Niue")
"NF":"Oceania", # "Norfolk Island")
"MP":"Oceania", # "Commonwealth of the Northern Mariana Islands")
"NO":"Europe", # "Kingdom of Norway")
"OM":"Asia", # "Sultanate of Oman")
"PK":"Asia", # "Islamic Republic of Pakistan")
"PW":"Oceania", # "Republic of Palau")
"PS":"Asia", # "Occupied Palestinian Territory")
"PA":"North America", # "Republic of Panama")
"PG":"Oceania", # "Independent State of Papua New Guinea")
"PY":"South America", # "Republic of Paraguay")
"PE":"South America", # "Republic of Peru")
"PH":"Asia", # "Republic of the Philippines")
"PN":"Oceania", # "Pitcairn Islands")
"PL":"Europe", # "Republic of Poland")
"PT":"Europe", # "Portuguese Republic")
"PR":"North America", # "Commonwealth of Puerto Rico")
"QA":"Asia", # "State of Qatar")
"RE":"Africa", # "Réunion")
"RO":"Europe", # "Romania")
"RU":"Europe", # "Russian Federation")
"RW":"Africa", # "Republic of Rwanda")
"BL":"North America", # "Saint Barthélemy")
"SH":"Africa", # '654'
"KN":"North America", # "Federation of Saint Kitts and Nevis")
"LC":"North America", # "Saint Lucia")
"MF":"North America", # "Saint Martin (French part)")
"PM":"North America", # "Saint Pierre and Miquelon")
"VC":"North America", # "Saint Vincent and the Grenadines")
"WS":"Oceania", # "Independent State of Samoa")
"SM":"Europe", # "Republic of San Marino")
"ST":"Africa", # "Democratic Republic of Sao Tome and Principe")
"SA":"Asia", # "Kingdom of Saudi Arabia")
"SN":"Africa", # "Republic of Senegal")
"RS":"Europe", # "Republic of Serbia")
"SC":"Africa", # "Republic of Seychelles")
"SL":"Africa", # "Republic of Sierra Leone")
"SG":"Asia", # "Republic of Singapore")
"SX":"North America", # "Sint Maarten (Dutch part)")
"SK":"Europe", # "Slovakia (Slovak Republic)")
"SI":"Europe", # "Republic of Slovenia")
"SB":"Oceania", # "Solomon Islands")
"SO":"Africa", # "Somali Republic")
"ZA":"Africa", # "Republic of South Africa")
"GS":"Antarctica", # "South Georgia and the South Sandwich Islands")
"SS":"Africa", # "Republic of South Sudan")
"ES":"Europe", # "Kingdom of Spain")
"LK":"Asia", # "Democratic Socialist Republic of Sri Lanka")
"SD":"Africa", # "Republic of Sudan")
"SR":"South America", # "Republic of Suriname")
"SJ":"Europe", # "Svalbard & Jan Mayen Islands")
"SZ":"Africa", # "Kingdom of Swaziland")
"SE":"Europe", # "Kingdom of Sweden")
"CH":"Europe", # "Swiss Confederation")
"SY":"Asia", # "Syrian Arab Republic")
"TW":"Asia", # "Taiwan
"TJ":"Asia", # "Republic of Tajikistan")
"TZ":"Africa", # "United Republic of Tanzania")
"TH":"Asia", # "Kingdom of Thailand")
"TL":"Asia", # "Democratic Republic of Timor-Leste")
"TG":"Africa", # "Togolese Republic")
"TK":"Oceania", # "Tokelau")
"TO":"Oceania", # "Kingdom of Tonga")
"TT":"North America", # "Republic of Trinidad and Tobago")
"TN":"Africa", # "Tunisian Republic")
"TR":"Asia", # "Republic of Turkey")
"TM":"Asia", # "Turkmenistan")
"TC":"North America", # "Turks and Caicos Islands")
"TV":"Oceania", # "Tuvalu")
"UG":"Africa", # "Republic of Uganda")
"UA":"Europe", # "Ukraine")
"AE":"Asia", # "United Arab Emirates")
"GB":"Europe", # "United Kingdom of Great Britain & Northern Ireland")
"US":"North America", # "United States of America")
"UM":"Oceania", # "United States Minor Outlying Islands")
"VI":"North America", # "United States Virgin Islands")
"UY":"South America", # "Eastern Republic of Uruguay")
"UZ":"Asia", # "Republic of Uzbekistan")
"VU":"Oceania", # "Republic of Vanuatu")
"VE":"South America", # "Bolivarian Republic of Venezuela")
"VN":"Asia", # "Socialist Republic of Vietnam")
"WF":"Oceania", # "Wallis and Futuna")
"EH":"Africa", # "Western Sahara")
"YE":"Asia", # "Yemen")
"ZM":"Africa", # "Republic of Zambia")
"ZW":"Africa" # "Republic of Zimbabwe");
}

mode = 'monthly' # options are "monthly" or "seasonal"
method = 'detrended_covar' # options are detrended/nondetrended_covar/svd
dtype = "snow"

data_path = "./Data/data/"
figs_path = "./figs/"+mode+"_figs/"+method+"/"
logs_path = "./logs/"+method+"/"+mode+"/"
tabl_path = "./tables/"+mode+"_tables/"+method+"/"


def load_latlon(cpath="./Data/lat_lon/", dim=(720, 720)):
    # loading data into dict
    coords = []
    for fname in os.listdir(cpath):
            if fname.endswith(".double"):
                print("Loading "+fname)
                with open(cpath+fname, 'rb') as f:  # open function read binary file
                    file = np.fromfile(f, dtype=float)  # Unsigned 8 bit Integer (0-2^7)
                    file = file.reshape(dim[0], dim[1]) # convert to matrix

                    coords.append(file)

    # lon is in [0], lat is in [1]
    return coords[1], coords[0]

def get_country(lats, lons):
    gmaps = googlemaps.Client(key=credentials.gmaps_key)

    ccode = []
    for lat, lon in zip(lats, lons):
        result = gmaps.reverse_geocode((lat, lon))
        # getting country code
        for res in result[0]['address_components']:
            if "country" in res['types']:
                ccode.append(res['short_name'])

    return ccode

def main():
    lat, lon = load_latlon()

    ## getting indicies of lat/lon that need to be used
    ## (in order to avoid making 720*720 google maps api calls)

    suffix = '1' # hardcoded for now but eventually will need to be fixed
    file = logs_path + dtype+"_stds_"+suffix+".txt"
    with open(file) as f:
        lines = [int(line.rstrip()) for line in f]

    # getting just the lattitude and longitudes that I need
    lat = lat.flatten()
    lon = lon.flatten()
    lats = lat[lines]
    lons = lon[lines]

    print("Getting countries...")
    country_codes = get_country(lats[:10], lons[:10])

    # add to pandas
    print("Creating DataFrame...")
    df = pd.DataFrame(columns=['lat_lon', 'country_code', 'continent'])
    for i in range(10): # testing
    # for i in range(len(lats)):
        # adding row to end of dataframe (there is definetly a better way to do this)
        df.loc[len(df.index)] = [(lats[i], lons[i]),
                                country_codes[i],
                                continent.get(country_codes[i])]


    # saving
    print("Saving DataFrame...")
    df.to_csv(logs_path + dtype+"_geo_"+suffix+".csv")



if __name__=="__main__":
    main()
