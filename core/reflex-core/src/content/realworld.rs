//! Generated real-world text: dates, times, prices, addresses, numbers. Gives the trainer
//! natural reasons to practise digits and symbols without random symbol soup.

use super::Lang;
use rand::Rng;
use rand::seq::IndexedRandom;

const NAMES: [&str; 12] =
    ["anna", "jonas", "lea", "max", "sara", "tim", "mia", "paul", "nina", "felix", "emma", "noah"];
const DOMAINS: [&str; 6] = ["example.com", "mail.de", "posteo.net", "web.de", "uni-berlin.de", "studio.io"];
const PATHS: [&str; 8] = ["docs", "blog", "help", "api/v2", "news", "shop/cart", "login", "about"];

pub fn real_world_line(lang: Lang, rng: &mut impl Rng) -> String {
    let day = rng.random_range(1..=28);
    let month = rng.random_range(1..=12);
    let year = rng.random_range(2019..=2030);
    let hour = rng.random_range(7..=21);
    let minute = [0, 15, 30, 45, 5, 50].choose(rng).copied().unwrap_or(0);
    let euros = rng.random_range(2..=499);
    let cents = [0, 49, 90, 95, 99, 50].choose(rng).copied().unwrap_or(0);
    let name = NAMES.choose(rng).unwrap();
    let other = NAMES.choose(rng).unwrap();
    let domain = DOMAINS.choose(rng).unwrap();
    let path = PATHS.choose(rng).unwrap();
    let count = rng.random_range(2..=48);
    let percent = rng.random_range(3..=95);
    let order = rng.random_range(1000..=99999);
    let room = rng.random_range(1..=420);
    let pick = rng.random_range(0..10);
    match lang {
        Lang::De => match pick {
            0 => format!("Termin am {day:02}.{month:02}.{year} um {hour}:{minute:02} Uhr in Raum {room}."),
            1 => format!("Der Preis liegt bei {euros},{cents:02} € inkl. MwSt. (Bestellung #{order})."),
            2 => format!("Schreib mir an {name}.{other}@{domain}, ich antworte bis {hour}:{minute:02}."),
            3 => format!("Infos unter https://{domain}/{path} - Stand: {day}.{month}.{year}"),
            4 => format!("Rabatt: {percent} % auf {count} Artikel, nur bis zum {day:02}.{month:02}.!"),
            5 => format!("Ergebnis: {count} + {percent} = {}, also rund {}0.", count + percent, (count + percent) / 10),
            6 => format!("Ruf an: +49 30 {}{} {order} (Mo-Fr, {hour}-{} Uhr).", room % 9 + 1, count, (hour + 8).min(22)),
            7 => format!("Ticket {order}/{year}: Status \"offen\", Prio {}, Team @{name}.", count % 3 + 1),
            8 => format!("Miete {euros}0 € + {count},{cents:02} € Nebenkosten = Summe?"),
            _ => format!("Lieferung ({count} Stk.) am {day}. {}; Gewicht ca. {percent} kg.", month_de(month)),
        },
        Lang::En => match pick {
            0 => format!("Meeting on {}/{day:02}/{year} at {}:{minute:02} {} in room {room}.", month, hour_12(hour), am_pm(hour)),
            1 => format!("Total: ${euros}.{cents:02} incl. tax (order #{order})."),
            2 => format!("Email me at {name}.{other}@{domain}, I'll reply by {}:{minute:02}.", hour_12(hour)),
            3 => format!("See https://{domain}/{path} for details - updated {month}/{day}/{year}."),
            4 => format!("Save {percent}% on {count} items, today only!"),
            5 => format!("Result: {count} * 2 = {}, and {percent} - {count} = {}.", count * 2, percent as i64 - count as i64),
            6 => format!("Call (555) {room:03}-{order:05} between {} and {} pm.", hour_12(hour), hour_12(hour + 2)),
            7 => format!("Ticket #{order}: status \"open\", priority {}, owner @{name}.", count % 3 + 1),
            8 => format!("Rent ${euros}0 + ${count}.{cents:02} utilities = what?"),
            _ => format!("Ship {count} units on {} {day}; weight approx. {percent} lb.", month_en(month)),
        },
    }
}

fn hour_12(h: u32) -> u32 {
    match h % 12 {
        0 => 12,
        x => x,
    }
}

fn am_pm(h: u32) -> &'static str {
    if h < 12 { "am" } else { "pm" }
}

fn month_de(m: u32) -> &'static str {
    ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]
        [(m as usize - 1) % 12]
}

fn month_en(m: u32) -> &'static str {
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        [(m as usize - 1) % 12]
}
