# vcrunch

## Scenario: encode with custom parameters
* Given an MP4 file "<src>"
* And an output directory "<out>"
* When I set VCRUNCH_INPUTS to "<src>"
* And I set VCRUNCH_TARGET_SIZE to "<size>"
* And I set VCRUNCH_AUDIO_BITRATE to "<audio>"
* And I set VCRUNCH_SAFETY_OVERHEAD to "<overhead>"
* And I set VCRUNCH_OUTPUT_DIR to "<out>"
* And I set VCRUNCH_MANIFEST_NAME to "<manifest>"
* And I set VCRUNCH_NAME_SUFFIX to "<suffix>"
* And I set VCRUNCH_SVT_LP to "<lp>"
* And I run vcrunch
* And vcrunch remuxes the encoded video with mkvmerge
* And mkvmerge reports version "95.0"
* Then vcrunch creates an AV1 file in "<out>"
* And the file name ends with "<suffix>.mkv"
* And "<manifest>" records "<src>" as done
* And the encode respects the target size
* And the encode respects the audio bitrate
* And the encode respects the safety overhead
* And the encode uses the SVT-AV1 lp "<lp>"

## Scenario: skip already encoded videos
* Given an MP4 file "<src>"
* And "<src>" already encoded into "<out>" with name ending "<suffix>.mkv"
* And a manifest "<manifest>" in "<out>" marking "<src>" as done
* When I set VCRUNCH_INPUTS to "<src>"
* And I set VCRUNCH_OUTPUT_DIR to "<out>"
* And I set VCRUNCH_MANIFEST_NAME to "<manifest>"
* And I set VCRUNCH_NAME_SUFFIX to "<suffix>"
* And I run vcrunch
* Then vcrunch skips "<src>"

## Scenario: read inputs from a list and filter by glob
* Given a list file "<list>" containing paths
* And some paths match "<pattern>" and others do not
* When I set VCRUNCH_PATHS_FROM to "<list>"
* And I set VCRUNCH_PATTERN to "<pattern>"
* And I run vcrunch
* Then matching video files are transcoded
* And non-matching paths are skipped

## Scenario: copy files when inputs fit target size
* Given an MP4 file "<src>"
* And an output directory "<out>"
* When I set VCRUNCH_INPUTS to "<src>"
* And I set VCRUNCH_TARGET_SIZE to "<size>"
* And I set VCRUNCH_OUTPUT_DIR to "<out>"
* And I run vcrunch
* Then "<src>" is copied to "<out>"
* And ".job.json" records "<src>" as done

## Scenario: move files when inputs fit target size
* Given an MP4 file "<src>"
* And an output directory "<out>"
* When I set VCRUNCH_INPUTS to "<src>"
* And I set VCRUNCH_TARGET_SIZE to "<size>"
* And I set VCRUNCH_OUTPUT_DIR to "<out>"
* And I set VCRUNCH_MOVE_IF_FIT to true
* And I run vcrunch
* Then "<src>" is moved to "<out>"
* And ".job.json" records "<src>" as done

## Scenario: ignore dot-underscore files
* Given a directory "<src>" containing a dot-underscore file "._clip.mov"
* And "<src>" also contains a video "<video>"
* And an output directory "<out>"
* When I set VCRUNCH_INPUTS to "<src>"
* And I set VCRUNCH_OUTPUT_DIR to "<out>"
* And I run vcrunch
* Then "._clip.mov" is ignored
* And "<video>" is processed

## Scenario: resume media probes
* Given a directory "<src>" containing a video "<video>" and a non-video "<asset>"
* And an output directory "<out>"
* When I set VCRUNCH_INPUTS to "<src>"
* And I set VCRUNCH_OUTPUT_DIR to "<out>"
* And I run vcrunch
* And I interrupt vcrunch after probe results are written
* And I run vcrunch again with the same environment
* Then vcrunch resumes using the existing probe entries
* And ".job.json" includes a "probes" entry for "<video>"
* And ".job.json" includes a "probes" entry for "<asset>"
