create table public.players (
  id uuid primary key default gen_random_uuid(),
  first_name text not null,
  last_name text not null,
  position text,
  team text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table public.games (
  id uuid primary key default gen_random_uuid(),
  game_date date not null,
  home_team text,
  away_team text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table public.pitches (
  id uuid primary key default gen_random_uuid(),
  game_id uuid not null references public.games(id) on delete cascade,
  batter_id uuid not null references public.players(id) on delete cascade,
  pitch_type text not null,
  speed numeric(4,1),
  description text,
  created_at timestamptz default now()
);

create index idx_pitches_game_id on public.pitches(game_id);
create index idx_pitches_batter_id on public.pitches(batter_id);
create index idx_games_date on public.games(game_date);
create index idx_players_team on public.players(team);
