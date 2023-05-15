import { Component, Input, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { Experiment } from 'src/app/entities/experiment';
import { Media } from 'src/app/entities/media';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
@Component({
  selector: 'app-discovery-vis',
  templateUrl: './discovery-vis.component.html',
  styleUrls: ['./discovery-vis.component.scss'],
})
export class DiscoveryVisComponent implements OnInit {
  @Input() experiment?: Experiment;
  @Input() mediaArray$?: Observable<Media[]>;
  mediaSrcArray$?: Observable<
    { seed: number; iteration: number; src: string }[]
  >;

  seed_collection?: number[];
  iteration_collection?: number[];

  constructor(private domSanitizer: DomSanitizer) {}
  ngOnInit(): void {
    // generates dummy_collections for *ngFor to loop over
    const nb_seeds = this.experiment?.config?.nb_seeds ?? 0;
    const nb_iterations = this.experiment?.config?.nb_iterations ?? 0;
    this.seed_collection = Array(nb_seeds);
    this.iteration_collection = Array(nb_iterations);

    this.mediaSrcArray$ = this.mediaArray$?.pipe(
      map((mediaArray) => {
        return mediaArray.map(this.generateSrc);
      })
    );
  }

  private generateSrc(media: Media): {
    seed: number;
    iteration: number;
    src: string;
  } {
    return {
      seed: media.seed,
      iteration: media.iteration,
      src: URL.createObjectURL(media.content),
    };
  }
}
